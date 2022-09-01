import os
import random
import torch
import torch.nn.functional as F
import numpy as np

import deepspeed

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from transformers import Trainer
from sofa.utils import print_rank_0, mpu, report_memory, save_checkpoint, Timers
from sofa.utils.dureader_eval import normalize
from sofa.utils.dureader_eval import compute_bleu_rouge

# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False

if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from sofa.distributed import DistributedDataParallel as DDP

class TrainerPlug(object):
    def __init__(self, model, optimizer, lr_scheduler, test_dataset, 
                 train_dataset, eval_dataset, pruner, args):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.pruner = pruner
        self.args = args

    def get_parameter_number(self, net):
        #for name, model in net.named_parameters():
        #    print_rank_0('name: {}, parameters: {}'.format(name, model.numel()))
        #exit()
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print_rank_0('Total: {}, Trainable: {}'.format(total_num, trainable_num))

    def get_batch(self, data_iterator, timers):
        ''' get_batch subdivides the source data into chunks of
        length args.seq_length. If source is equal to the example
        output of the data loading example, with a seq_length limit
        of 2, we'd get the following two Variables for i = 0:
        ┌ a g m s ┐ ┌ b h n t ┐
        └ b h n t ┘ └ c i o u ┘
        Note that despite the name of the function, the subdivison of data is not
        done along the batch dimension (i.e. dimension 1), since that was handled
        by the data loader. The chunks are along dimension 0, corresponding
        to the seq_len dimension in the LSTM. A Variable representing an appropriate
        shard reset mask of the same dimensions is also returned.
        '''
        # Items and their type.
        keys = ['input_ids', 'input_mask', 'segment_ids', 'label_id']
        datatype = torch.int64

        # Broadcast data.
        timers('data loader').start()
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        timers('data loader').stop()
        if data is not None:
            for key in data:
                if type(data[key]) == list:
                    data[key] = torch.stack(data[key], 0)
                    data[key] = data[key].transpose(0, 1)
        data_b = mpu.broadcast_data(keys, data, datatype)
        # Unpack.
        tokens = data_b['input_ids'].long()
        types = data_b['segment_ids'].long()
        label_id = data_b['label_id'].long()
        padding_mask = data_b['input_mask'].byte()
        return tokens, types, label_id, padding_mask
    
    def forward_step(self, iteration, data_iterator, model, args, timers, checkpoint_activations=False):
        """Forward step."""

        # Get the batch.
        timers('batch generator').start()
        tokens, types, label_id, padding_mask = self.get_batch(data_iterator, timers)
        #if iteration < args.batch_warmup * args.train_iters and iteration != -1:
        #    index = multi8(iteration / (args.batch_warmup * args.train_iters) * args.batch_size)
        #    tokens = tokens[:index]
        #    types = types[:index]
        #    next_sentence = next_sentence[:index]
        #    loss_mask = loss_mask[:index]
        #    lm_labels = lm_labels[:index]
        #    padding_mask = padding_mask[:index]
        timers('batch generator').stop()
        
        # Forward model.
        timers('fwd step1').start()
        cls = self.model(tokens, types, padding_mask,
                            checkpoint_activations=checkpoint_activations,
                            detach_index=args.detach_index)
        timers('fwd step1').stop()

        timers('fwd step2').start()
        cls_loss = F.cross_entropy(cls.view(-1, args.num_of_classes).contiguous().float(),
                                label_id.view(-1).contiguous(),
                                ignore_index=-1)
        timers('fwd step2').stop()

        timers('fwd step3').start()
        right = torch.sum((torch.max(cls.view(-1, args.num_of_classes).contiguous(), -1, keepdim=True)[1] == label_id).float())
        if mpu.get_model_parallel_rank() == 0:
            print('Get acc:' ,model.training, iteration, right / label_id.size(0))
        timers('fwd step3').stop()

        return cls, cls_loss, right

    def backward_step(self, optimizer, model, cls_loss, args, timers):
        """Backward step."""

        # Total loss.
        loss = cls_loss

        # Backward pass.
        if args.deepspeed: 
            model.backward(loss)
        else:
            optimizer.zero_grad()
            if args.fp16:
                optimizer.backward(loss, update_master_grads=False)
            else:
                loss.backward()
        # Reduce across processes.
        #lm_loss_reduced = lm_loss
        #nsp_loss_reduced = nsp_loss

        #reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1)))
        reduced_losses = loss.view(1)
    
        if args.deepspeed: 
            # DeepSpeed backward propagation already addressed all reduce communication. 
            # Reset the timer to avoid breaking timer logs below. 
            timers('allreduce').reset()
        else:
            torch.distributed.all_reduce(reduced_losses.data)
            reduced_losses.data = reduced_losses.data / args.world_size
            if not USE_TORCH_DDP:
                timers('allreduce').start() 
                model.allreduce_params(reduce_after=False,
                                    fp32_allreduce=args.fp32_allreduce)
                timers('allreduce').stop() 

        # Update master gradients.
        if not args.deepspeed:
            if args.fp16:
                optimizer.update_master_grads()

            # Clipping gradients helps prevent the exploding gradient.
            if args.clip_grad > 0:
                if not args.fp16:
                    mpu.clip_grad_norm(model.parameters(), args.clip_grad)
                else:
                    optimizer.clip_master_grads(args.clip_grad)

        return cls_loss

    def train_step(self, iteration, data_iterator, model, optimizer, lr_scheduler,
                args, timers):
        """Single training step."""
        # Forward model for one step.
        timers('Forward').start()
        _, cls_loss, _ = self.forward_step(iteration, data_iterator, model,
                                        args, timers, checkpoint_activations=args.checkpoint_activations)
        timers('Forward').stop()

        # Calculate gradients, reduce across processes, and clip.
        timers('Backward').start()
        cls_loss_reduced = self.backward_step(optimizer, model, cls_loss, args, timers)
        timers('Backward').stop()

        # Update parameters.
        skipped_iter = 0
        timers('Optimizer').start()
        print_rank_0(model.is_gradient_accumulation_boundary())
        if args.deepspeed:
            model.step()
        else:
            optimizer.step()

            # Update learning rate.
            if not (args.fp16 and optimizer.overflow):
                lr_scheduler.step()
            else:
                skipped_iter = 1
        timers('Optimizer').stop()

        return cls_loss_reduced, skipped_iter

    def _train(self, model, optimizer, lr_scheduler,
            train_data_iterator, val_data, timers, args):
        """Train the model."""

        # Turn on training mode which enables dropout.
        model.train()

        # Tracking loss.
        total_cls_loss = 0.0

        # Iterations.
        iteration = args.iteration
        cur_iteration = 0
        skipped_iters = 0

        report_memory_flag = True
        world_size = torch.distributed.get_world_size(
            group=mpu.get_data_parallel_group())
        epoch_iters = args.data_size[0].item() // (world_size * args.batch_size) 
        while cur_iteration < epoch_iters:
            cls_loss, skipped_iter = self.train_step(iteration,
                                                        train_data_iterator,
                                                        model,
                                                        optimizer,
                                                        lr_scheduler,
                                                        args, timers)
            skipped_iters += skipped_iter
            iteration += 1
            cur_iteration += 1
            # Update losses.
            total_cls_loss += cls_loss.data.detach().float()

            # Logging.
            print_rank_0('global_step={}'.format(iteration))
            if iteration % args.log_interval == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                avg_cls_loss = total_cls_loss.item() / args.log_interval
                elapsed_time = timers('interval time').elapsed()
                log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                                args.train_iters)
                log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                    elapsed_time * 1000.0 / args.log_interval)
                log_string += ' learning rate {:.3E} |'.format(learning_rate)
                log_string += ' lm loss {:.6E} |'.format(avg_cls_loss)
                if args.fp16:
                    log_string += ' loss scale {:.1f} |'.format(
                        optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
                print_rank_0(log_string)
                print_rank_0('Memory allocated: {} GB'.format(torch.cuda.memory_allocated() / 2**30))
                print_rank_0('Max memory allocated: {} GB'.format(torch.cuda.max_memory_allocated() / 2**30))
                print_rank_0('Cache allocated: {} GB'.format(torch.cuda.memory_cached() / 2**30))
                print_rank_0('Max cache allocated: {} GB'.format(torch.cuda.max_memory_cached() / 2**30))
                total_cls_loss = 0.0
                if report_memory_flag:
                    report_memory('after {} iterations'.format(iteration))
                    report_memory_flag = False
                timers.log(['Forward', 'fwd step1', 'fwd step2', 'fwd step3', 'Backward', 'Optimizer', 'batch generator',
                            'data loader'],
                        normalizer=args.log_interval)
            # Checkpointing
            #if args.save and args.save_interval and iteration % args.save_interval == 0:
            #    save_checkpoint(iteration, model, optimizer, lr_scheduler, args)
            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
                if val_data is not None:
                    val_data_iterator = iter(val_data)
                else:
                    val_data_iterator = None
                prefix = 'iteration {}'.format(iteration)
                val_loss, _ = self.evaluate_and_print_results(prefix, val_data_iterator,
                                                        model, args, timers, 'dev', False)
            #    evaluate_and_print_results(
            #        prefix, val_data_iterator, model, args, timers, False)

            #if args.exit_interval and iteration % args.exit_interval == 0:
            #    torch.distributed.barrier()
            #    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #    rank = torch.distributed.get_rank()
            #    print('rank: {} | time: {} | exiting the program at iteration {}'.
            #          format(rank, time_str, iteration), flush=True)
            #    exit()

        return iteration, skipped_iters

    def evaluate(self, data_iterator, model, args, timers, set_type, verbose = False):
        """Evaluation."""

        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        total_cls_loss = 0
        total_right = 0
        if verbose and torch.distributed.get_rank() % 16 == 0:
            os.makedirs(args.save, exist_ok=True)
            fout = open(args.save+'/'+str(args.iteration)+'_'+str(torch.distributed.get_rank()), 'w')
        with torch.no_grad():
            iteration = 0
            world_size = torch.distributed.get_world_size(
                group=mpu.get_data_parallel_group())
            if args.data_size[1 if set_type=='dev' else 2].item() % (world_size * args.eval_batch_size) != 0 and set_type != 'dev':
                eval_iters = args.data_size[1 if set_type=='dev' else 2].item() // (world_size * args.eval_batch_size) + 1
            else:
                eval_iters = args.data_size[1 if set_type=='dev' else 2].item() // (world_size * args.eval_batch_size)
            while iteration < eval_iters:
                iteration += 1
                if verbose and iteration % args.log_interval == 0:
                    print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
                # Forward evaluation.
                cls_logits, cls_loss, right = self.forward_step(-1, data_iterator, self.model,
                                                args, timers, checkpoint_activations=False)
                if verbose and torch.distributed.get_rank() % 16 == 0:
                    cls_loss_ = cls_logits.detach().cpu().numpy().tolist()
                    for item in cls_loss_:
                        fout.write(str(np.argmax(item))+'\t'+str(item)+'\n')
                        fout.flush()
                torch.distributed.all_reduce(right,
                            op=torch.distributed.ReduceOp.SUM)
                right = right / torch.distributed.get_world_size(group=mpu.get_model_parallel_group())
                # Reduce across processes.
                if isinstance(self.model, DDP):
                    reduced_losses = cls_loss
                    torch.distributed.all_reduce(reduced_losses.data)
                    reduced_losses.data = reduced_losses.data/args.world_size
                    cls_loss = reduced_losses

                total_cls_loss += cls_loss.data.detach().float().item()
                total_right += right.data.detach().item()
        # Move model back to the train mode.
        self.model.train()

        total_cls_loss /= eval_iters
        total_right /= eval_iters * world_size * args.eval_batch_size
        return total_cls_loss, total_right

    def evaluate_and_print_results(self,prefix, data_iterator, model,
                                args, timers, set_type, verbose=False):
        """Helper function to evaluate and dump results on screen."""
        cls_loss, cls_acc = self.evaluate(data_iterator, self.model,
                                    args, timers, set_type, verbose)
        print_rank_0('-' * 100)
        string = ' validation loss at {} | '.format(prefix)
        string += 'CLS loss: {:.6E} | '.format(cls_loss)
        string += 'CLS acc: {:.6f} | '.format(cls_acc)
        length = len(string) + 1
        print_rank_0('-' * length)
        print_rank_0(string)
        print_rank_0('-' * length)

        return cls_loss, None

    def train(self):
        args = self.args
        task_name = args.task_name.lower()
        #Timer
        timers = Timers()
                
        self.get_parameter_number(self.model)

        timers('interval time').start()
        for epoch_id in range(args.num_epochs):
            print_rank_0('Training on epoch {}'.format(epoch_id))
            args.epoch_id = epoch_id
            if self.train_dataset is not None:
                train_data_iterator = iter(self.train_dataset)
            else:
                train_data_iterator = None
            if self.eval_dataset is not None:
                val_data_iterator = iter(self.eval_dataset)
            else:
                val_data_iterator = None
            """ 
            if args.do_valid:
                prefix = 'the end of training for val data'
                val_loss, _ = self.evaluate_and_print_results(prefix, val_data_iterator,
                                                        self.model, args, timers, 'dev', False)
            """
            if args.train_iters > 0:
                if args.do_train:
                    args.iteration, skipped = self._train(self.model, self.optimizer,
                                            self.lr_scheduler,
                                            train_data_iterator,
                                            self.eval_dataset,
                                            timers, args)
                if args.do_valid:
                    prefix = 'the end of training for val data'
                    val_loss, _ = self.evaluate_and_print_results(prefix, val_data_iterator,
                                                        self.model, args, timers, 'dev', False)
            if args.save and args.iteration != 0 and epoch_id == args.num_epochs - 1:
                try:
                    # save_checkpoint(args.iteration, self.model, self.optimizer, self.lr_scheduler, self.pruner, args)
                    print_rank_0('=== Final: Begin to saved model at iteration: {} ==='.format(args.iteration))
                    save_checkpoint(args.iteration, self.model, self.optimizer, self.lr_scheduler, args)
                    print_rank_0('=== Final: Successfully saved model at iteration: {} ==='.format(args.iteration))
                except Exception as e:
                    print('Error in saved model at end of training', flush=True)
                    print(e, flush=True)

            '''
            if self.test_dataset is not None:
                test_data_iterator = iter(self.test_dataset)
            else:
                test_data_iterator = None
        
            if args.do_test:
                # Run on test data.
                prefix = 'the end of training for test data'
                self.evaluate_and_print_results(prefix, test_data_iterator,
                                        self.model, args, timers, 'test', True)
            '''

class TrainerPlugNLG(TrainerPlug):
    def __init__(self, model, optimizer, lr_scheduler, test_dataset, 
                 train_dataset, eval_dataset, pruner, tokenizer, args):
        super().__init__(model, optimizer, lr_scheduler, test_dataset, 
                         train_dataset, eval_dataset, pruner, args)
        self.tokenizer = tokenizer
    
    def forward_step(self, iteration, data_iterator, model, args, timers, checkpoint_activations=False):
        # Get the batch.
        timers('batch generator').start()
        tokens, types, padding_mask, target_tokens, target_labels, dec_loss_mask, attention_mask, position_ids = self.get_batch(data_iterator, args, timers)
        #torch.distributed.barrier()
        #print ("\n\n\n")
        timers('batch generator').stop()
        # Forward model.
        timers('fwd step1').start()
        prediction_scores, output = model(tokens, types, padding_mask, target_tokens, position_ids, attention_mask,
                        checkpoint_activations=args.checkpoint_activations)
        timers('fwd step1').stop()

        timers('fwd step2').start()
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(),
                                                target_labels)
        #print_rank_0(torch.sum((torch.max(output.contiguous(), 2)[1] == target_labels).float()* loss_mask.float())/loss_mask.sum())
        dec_loss_mask = dec_loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * dec_loss_mask) / dec_loss_mask.sum()
        
        timers('fwd step2').stop()
        
        timers('fwd step3').start()
        timers('fwd step3').stop()
        return None, loss, None
    
    def get_batch(self, data_iterator, args, timers):
        ''' get_batch subdivides the source data into chunks of
        length args.seq_length. If source is equal to the example
        output of the data loading example, with a seq_length limit
        of 2, we'd get the following two Variables for i = 0:
        ┌ a g m s ┐ ┌ b h n t ┐
        └ b h n t ┘ └ c i o u ┘
        Note that despite the name of the function, the subdivison of data is not
        done along the batch dimension (i.e. dimension 1), since that was handled
        by the data loader. The chunks are along dimension 0, corresponding
        to the seq_len dimension in the LSTM. A Variable representing an appropriate
        shard reset mask of the same dimensions is also returned.
        '''
        # Items and their type.
        keys = ['input_ids', 'input_mask', 'segment_ids', 'target_ids']
        datatype = torch.int64
        # iteration -> bucket
        # Broadcast data.
        timers('data loader').start()
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        if data is not None:
            for key in data:
                if type(data[key]) == list:
                    data[key] = torch.stack(data[key], 0)
                    data[key] = data[key].transpose(0, 1)
        timers('data loader').stop()
        data_b = mpu.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens = data_b['input_ids'].long()
        types = data_b['segment_ids'].long()
        padding_mask = data_b['input_mask'].byte()
        target_ids = data_b['target_ids'].long()
        #
        
        target_tokens = target_ids[:, :-1].contiguous()
        target_labels = target_ids[:, 1:].contiguous()

        # Get the masks and postition ids.
        
        attention_mask, dec_loss_mask, position_ids = self.get_masks_and_position_ids(
            target_tokens,
            0)
        if args.fp16:
            attention_mask = attention_mask.half()

        return tokens, types, padding_mask, target_tokens, target_labels, dec_loss_mask, attention_mask, position_ids

    def get_masks_and_position_ids(self, data,
                                eod_token,
                                reset_position_ids=False,
                                reset_attention_mask=False):
        # Extract batch size and sequence length.
        batch_size, seq_length = data.size()

        # Attention mask (lower triangular).
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device)).view(
                att_mask_batch, 1, seq_length, seq_length)

        # Loss mask.
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
        loss_mask[data == eod_token] = 0.0

        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i+1):, :(i+1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i+1):] -= (i + 1 - prev_index)
                        prev_index = i + 1

        return attention_mask, loss_mask, position_ids

    def evaluate(self, data_iterator, model, args, timers, set_type, verbose = False, tokenizer=None, log=False):
        """Evaluation."""
        tokenizer = self.tokenizer
        # Turn on evaluation mode which disables dropout.
        model.eval()

        total_cls_loss = 0
        total_right = 0
        temperature = 0.9
        top_k = 20
        top_p = 0.0
        pred_dict = {}
        ref_dict = {}
        counter_all = 0
        with torch.no_grad():
            iteration = 0
            world_size = torch.distributed.get_world_size(
                group=mpu.get_data_parallel_group())
            eval_iters = args.data_size[1 if set_type=='dev' else 2].item() // (world_size * args.eval_batch_size)
            while iteration < eval_iters:
                iteration += 1
                if verbose and iteration % args.log_interval == 0:
                    print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
                # Forward evaluation.
                tokens, types, padding_mask, target_tokens, target_labels, dec_loss_mask, attention_mask, position_ids = self.get_batch(data_iterator, args, timers)
                counter = 0
                cls_id = 101 
                vocab_size = 21127
                batch_size = tokens.size(0)
                device = tokens.device
                dec_input_ids = torch.full([batch_size, 1], cls_id, dtype=torch.long, device=device)
                sequence_output = None
                while counter < args.tgt_length:
                    position_ids = torch.full([batch_size, 1], counter, dtype=torch.long, device=device)
                    prediction_scores, output, sequence_output = model(tokens, types, padding_mask, dec_input_ids, position_ids, attention_mask,
                        checkpoint_activations=False, is_infer=True, parallel_output=False, sequence_output=sequence_output)
                    try:
                        logits = output[:, -1, :]
                        logits = logits / temperature
                        logits = self.top_k_logits(logits, top_k=top_k, top_p=top_p)            
                        log_probs = F.softmax(logits, dim=-1)
                        # prev = torch.multinomial(log_probs, num_samples=1)
                        prev = torch.argmax(log_probs, 1).unsqueeze(1)
                        dec_input_ids = torch.cat([dec_input_ids, prev], dim=1)
                        counter += 1
                    except:
                        break
                if mpu.get_model_parallel_rank() == 0:
                    target_list = target_labels.cpu().numpy().tolist()
                    pred_list = dec_input_ids.cpu().numpy().tolist()
                    for i in range(batch_size):
                        for j in range(len(pred_list[i])):
                            if pred_list[i][j] > vocab_size-1:
                                pred_list[i][j] = 100
                        gold = tokenizer.convert_ids_to_tokens(target_list[i])
                        pred = tokenizer.convert_ids_to_tokens(pred_list[i])
                        gold_string = "".join(gold).replace("##", "").split("[SEP]")[0].replace("[CLS]", "").replace("[SEP]", "").replace("[UNK]", "")
                        pred_string = "".join(pred).replace("##", "").split("[SEP]")[0].replace("[CLS]", "").replace("[SEP]", "").replace("[UNK]", "")
                        pred_dict[str(counter_all)] = normalize([pred_string])
                        ref_dict[str(counter_all)] = normalize([gold_string])
                        counter_all += 1
                    if log:
                        print("pred:", pred_string)
                        print("label:", gold_string)
                        print("input_ids:,", tokens[i].cpu().numpy().tolist())
                        print("outputs_ids:,", pred_list[i])
                        print("-"*100)
        if mpu.get_model_parallel_rank() == 0:
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            print("test result %d: " % counter_all, bleu_rouge)
                # Reduce across processes.
        return total_cls_loss

    def top_k_logits(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        # This function has been mostly taken from huggingface conversational ai code at
        # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
            
        if top_p > 0.0:
            #convert to 1D
            logits=logits.view(logits.size()[1]).contiguous()
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
            #going back to 2D
            logits=logits.view(1, -1).contiguous()

        return logits

    def evaluate_beam(self, data_iterator, model, args, timers, set_type, verbose = False, tokenizer=None):
        """Evaluation."""
        tokenizer = self.tokenizer

        # Turn on evaluation mode which disables dropout.
        model.eval()

        total_cls_loss = 0
        total_right = 0
        pred_dict = {}
        ref_dict = {}
        vocab_size = 21127
        beam_generator = TextGenerator(args, model, tokenizer, None) 
        counter_all = 0

        with torch.no_grad():
            iteration = 0
            world_size = torch.distributed.get_world_size(
                group=mpu.get_data_parallel_group())
            eval_iters = args.data_size[1 if set_type=='dev' else 2].item() // (world_size * args.eval_batch_size)
            while iteration < eval_iters:
                iteration += 1
                if verbose and iteration % args.log_interval == 0:
                    print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
                # Forward evaluation.
                tokens, types, padding_mask, target_tokens, target_labels, dec_loss_mask, attention_mask, position_ids = self.get_batch(data_iterator, args, timers)
                batch_size = tokens.size(0)
                # sequence_output = self.model.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=False, checkpoint_activations=args.checkpoint_activations)
                encoder_inputs = [tokens, types, padding_mask]
                
                result_dict = beam_generator.translate_batch(encoder_inputs)
                pred_list = result_dict["predictions"]
                target_list = target_labels.cpu().numpy().tolist()
                if mpu.get_model_parallel_rank() == 0:
                    for i in range(batch_size):
                        pred_ids = pred_list[i][0].cpu().numpy().tolist()
                        for j in range(len(pred_ids)):
                            if pred_ids[j] > vocab_size-1:
                                pred_ids[j] = 100
                        gold = tokenizer.convert_ids_to_tokens(target_list[i])
                        pred = tokenizer.convert_ids_to_tokens(pred_ids)
                        gold_string = "".join(gold).replace("##", "").split("[SEP]")[0].replace("[CLS]", "").replace("[SEP]", "").replace("[UNK]", "")
                        pred_string = "".join(pred).replace("##", "").split("[SEP]")[0].replace("[CLS]", "").replace("[SEP]", "").replace("[UNK]", "")
                        if len(gold_string) < 3:
                            continue
                        pred_dict[str(counter_all)] = normalize([pred_string])
                        ref_dict[str(counter_all)] = normalize([gold_string])
                        counter_all += 1
        if mpu.get_model_parallel_rank() == 0:
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            print (ref_dict["0"])
            print (pred_dict["0"])
            print("test result %d: " % counter_all, bleu_rouge)
                    
                # Reduce across processes.
        return total_cls_loss 

    def evaluate_and_print_results(self, prefix, data_iterator, model,
                                args, timers, set_type, verbose=False, tokenizer=None, log=False):
        """Helper function to evaluate and dump results on screen."""
        bleu = None
        if args.sample_topk:
            cls_loss = self.evaluate(data_iterator, model, args, timers, set_type, verbose, tokenizer=tokenizer, log=log)
        else:
            cls_loss = self.evaluate_beam(data_iterator, model,
                                        args, timers, set_type, verbose, tokenizer=tokenizer)
        print_rank_0('-' * 100)
        string = ' validation loss at {} | '.format(prefix)
        string += 'CLS loss: {:.6E} | '.format(cls_loss)
        length = len(string) + 1
        print_rank_0('-' * length)
        print_rank_0(string)
        print_rank_0('-' * length)

        return cls_loss, bleu

    
class TextGenerator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.alpha = 0.6

        self.logger = logger
        # self.cuda = args.visible_gpus != '-1'
        self.cuda = (torch.cuda.device_count() > 0)

        self.args = args
        self.model = model
        #TODO  generator
        #self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = 101 #['[PAD]']
        self.end_token = 102 #'[PAD]']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.tgt_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None



        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def tile(self, x, count, dim=0):
        """
        Tiles x on dimension dim count times.
        """
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x

    
    def translate_batch(self, encoder_inputs, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(encoder_inputs, self.max_length, min_length=self.min_length)

    def _fast_translate_batch(self,
                              encoder_inputs,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        tokens, types, padding_mask = encoder_inputs
        batch_size = tokens.size(0)
        device = tokens.device
        tmp_alive_seq = torch.full(
            [batch_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)
        prediction_scores, dec_feat_seq, sequence_output = self.model(tokens, types, padding_mask, tmp_alive_seq, None, None, checkpoint_activations=False, is_infer=True, parallel_output=False, sequence_output=None)
        src_features = sequence_output


        # Tile states and memory beam_size times.
        # dec_states.map_batch_fn(
        #     lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = self.tile(src_features, beam_size, dim=0)
        attention_mask = self.tile(padding_mask, beam_size, dim=0)
        #TODO support p_gen ...
        # if self.args.p_gen:
        #     src = tile(batch.src, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = []
        dec_attn_mask = None
        dec_position_ids = None

        for step in range(max_length):
            tgt_len = alive_seq.size()[1]
            #modified with tgt_len
            #repeat_attention_mask = attention_mask.repeat([1, 1, tgt_len, 1])
            #dec_attn_mask = _make_causal_mask(alive_seq.shape, attention_mask.dtype, device=alive_seq.device)
            #dec_feat_seq = self.model.decode(self.model.bert.embeddings, src_features, alive_seq,
            #                           enc_attn_mask=repeat_attention_mask, dec_attn_mask=dec_attn_mask)
            
            prediction_scores, dec_feat_seq, _ = self.model(tokens, types, attention_mask, alive_seq, dec_position_ids, dec_attn_mask, checkpoint_activations=False, is_infer=True, parallel_output=False, sequence_output=src_features)

            dec_feat_seq = dec_feat_seq[:, -1, :]
            vocab_size = dec_feat_seq.size(-1)
            log_probs = torch.log(torch.softmax(dec_feat_seq.view(-1, vocab_size), dim=-1))

            if step < min_length:
               log_probs[:, self.end_token] = -1e20
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.alpha #global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores = log_probs / length_penalty

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size, rounding_mode='trunc')
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1) #self.end_token)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1) #self.end_token)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1) #self.end_token)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        # if self.args.dataset == "qg_ranking_test" or (self.args.dataset == 'paraphrase' and (not self.args.sample_topk)):
                        #     for each in best_hyp[:beam_size]:
                        #         score, pred = each
                        #         results["scores"][b].append(score)
                        #         results["predictions"][b].append(pred)
                        # else:
                        score, pred = best_hyp[0]
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            attention_mask = attention_mask.index_select(0, select_indices)

        return results

