from model import Model
import torch
import argparse
def read_vocab(file):
    with open(file,'rt') as F:
        s = F.read()[1:-1]
    key_value = [kv.split(':') for kv in s.split(',')]
    key_value = [(str.strip(k),str.strip(v)) for (k,v) in key_value]
    key_value = dict([(k[1:-1],int(v)) for (k,v) in key_value])
    return key_value

def py_to_han(model,s,py_vocab,han_vocab_itos):
    s  = '<sos> ' + s + ' <eos>'
    x = []
    for _s in s.split(' '):
        if _s in py_vocab.keys():
            x.append(py_vocab[_s])
        else:
            x.append(py_vocab['<unk>'])
    
    x = torch.tensor(x).cuda()
    x= torch.unsqueeze(x,0)
    
    with torch.no_grad():
        model.eval()
        y = model(x)
    idx = torch.argmax(y[0].cpu(),1)
    h = []
    for i in idx:
        h.append(han_vocab_itos[int(i)])
    return ''.join(h[1:-1])

#def main():
    
parser = argparse.ArgumentParser()
parser.add_argument('--py_vocab', default='./data/py_vocab_sd.txt', 
                    type=str, required=False)
parser.add_argument('--han_vocab', default='./data/han_vocab_sd.txt', 
                    type=str, required=False)

parser.add_argument('--model_weight', default='./models/py2han_sd_model_epoch8val_acc0.957.pth', 
                    type=str, required=False)
parser.add_argument('--test_file', default='./data/ai_shell_test_sd.pinyin', 
                    type=str, required=False)


args = parser.parse_args([])
print('args:\n' + args.__repr__())



han_vocab = read_vocab(args.han_vocab)
py_vocab = read_vocab(args.py_vocab)

han_vocab_itos = dict([(han_vocab[k],k) for k in han_vocab.keys()])
py_vocab_itos = dict([(py_vocab[k],k) for k in py_vocab.keys()])
py_vocab_size = len(py_vocab)
ch_vocab_size = len(han_vocab)
py_vocab_size,ch_vocab_size
sd = torch.load(args.model_weight)
emb_dim = 512
hidden_dim = 512
n_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(py_vocab_size, emb_dim, hidden_dim,ch_vocab_size, n_layers).to(device)
model.load_state_dict(sd)
test_file = args.test_file
lines = open(test_file).read().split('\n')
for l in lines:
    print(l+'\n')
    ch = py_to_han(model,l,py_vocab,han_vocab_itos)
    print(ch+'\n')

# if __name__ == '__main__':
#     main()    
