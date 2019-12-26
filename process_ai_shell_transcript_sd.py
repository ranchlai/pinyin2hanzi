import pypinyin
import numpy as np
import argparse
np.random.seed(666)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='./data/aishell_transcript_v0.8.txt', 
                        type=str, required=False, help='AI shell transcript')
    args = parser.parse_args([])
    print('args:\n' + args.__repr__())

    file = args.file
    print('processing ',file)
    
    lines = open(file).read().split('\n')[:-1]
    lines = [l.split(' ')[1:] for l in lines]
    all_py = []
    for l in lines:
        all_py.append(' '.join([p[0] for p in pypinyin.pinyin(l)]))

    n = len(lines)
    idx = np.random.permutation(n)
    lines = [''.join(lines[i]) for i in idx]
    all_py = [all_py[i] for i in idx]    

    n_train = int(n*0.7)
    n_dev = int(n*0.9)
    with open('./data/ai_shell_train_sd.han','wt') as F:
        for l in lines[:n_train]:
            F.write(l+'\n')

    with open('./data/ai_shell_train_sd.pinyin','wt') as F:
        for l in all_py[:n_train]:
            F.write(l+'\n')
    with open('./data/ai_shell_dev_sd.han','wt') as F:
        for l in lines[n_train:n_dev]:
            F.write(l+'\n')

    with open('./data/ai_shell_dev_sd.pinyin','wt') as F:
        for l in all_py[n_train:n_dev]:
            F.write(l+'\n')
            
            
    with open('./data/ai_shell_test_sd.han','wt') as F:
        for l in lines[n_dev:]:
            F.write(l+'\n')

    with open('./data/ai_shell_test_sd.pinyin','wt') as F:
        for l in all_py[n_dev:]:
            F.write(l+'\n')
    print('{} lines processed,data saved to ./data/ai_shell_'.format(len(lines)))
    

if __name__ == '__main__':
    main()
