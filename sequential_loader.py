
def get_batch(split='train', single_pass=False):
    if split == 'train':
        data_path = os.path.join(data_dir, f'train.bin')
    elif split == 'val':
        data_path = os.path.join(data_dir, f'val.bin')
    elif split == 'test':
        data_path = os.path.join(data_dir, f'test.bin')
    else:
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split}")

    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    total_elements = len(data) - 1

    stride_length = int(block_size)

    indices = np.arange(0, total_elements - block_size, stride_length)
    #print0(f"Total elements: {total_elements}, indices: {len(indices)}")

    # Uncomment if you want to shuffle first epoch batches
    #np.random.shuffle(indices)

    # uncomment if you want to see how many iters per epoch
    #if split == 'train':
        #global iters_per_epoch
        #iters_per_epoch = int((len(indices) - batch_size + 1) / (batch_size * gradient_accumulation_steps))
        #print0('iters per epoch:', iters_per_epoch)
        #print0('iters for 150 epochs:', iters_per_epoch * 150)

    global epoch
    while True:
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            x_batch = []
            y_batch = []
            for j in range(batch_size):
                start_idx = indices[i + j]
                if start_idx + block_size + 1 > total_elements:
                    continue
                x = torch.from_numpy(data[start_idx:start_idx + block_size].astype(np.int64))
                y = torch.from_numpy(data[start_idx + 1:start_idx + block_size + 1].astype(np.int64))
                x_batch.append(x)
                y_batch.append(y)
            
            if len(x_batch) == batch_size:
                x_batch = torch.stack(x_batch).to(device)
                y_batch = torch.stack(y_batch).to(device)
                yield x_batch, y_batch
        
        # logging
        if split == 'train':
            epoch += 1
            print0(f"Finished epoch: {epoch}")
            if wandb_log:
                wandb.log({'iter': iter_num, 'epoch': epoch})
        
        # for eval, this stops after one epoch
        if single_pass:
            break

        # uncomment if you want to shuffle subsequent epochs
        #np.random.shuffle(indices)