from fairseq.models.transformer_lm import TransformerLanguageModel
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--lang_id",
        type=int,
        help="Directory to the dataset.",
        default=0,
    )
    
    args = parser.parse_args()
    return args

def bind(p, h, m):
    p1 = ". "
    p2 = ". "
    p3 = ', '
    p4 = ''
    return p + p1 + m + p3 + h

class Model:
    def __init__(self, model_dir = '/work/dlhlp2022/HW5/Assignment-cross-lingual-transfer/7.5B'
):
        self.lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')
        self.lm = self.lm.eval()
        self.lm = self.lm.half()
        self.lm = self.lm.cuda()
        self.pred_label = np.array(['entailment', 'contradiction', 'neutral'])
    def get_logprobs(self, prompt):
        import re
        prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines, which indicate separate documents
        return self.lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']

    def XNLI_eval(self, premise, hypothesis, prompts):
        p1 = ". "
        p2 = ", "
        lprob = {}
        for k in prompts.keys():
            if k == 'R':
                continue
            # lprob[k] = self.get_logprobs(premise + p1 + prompts['R'] + prompts[k] + p2 + hypothesis).sum().cpu()
            evaler = bind(premise, hypothesis, prompts['R'] + p1 + prompts[k])
            # print('51', evaler)
            lprob[k] = self.get_logprobs(evaler).sum().cpu()
        return max(lprob, key=lprob.get)
        
    def val_all(self, premise, hypothesis):
        lprob1 = self.get_logprobs(premise + " , right? Yes, " + hypothesis).sum().cpu()
        lprob2 = self.get_logprobs(premise + " , right? No, " + hypothesis).sum().cpu()
        lprob3 = self.get_logprobs(premise + " , right? Also, " + hypothesis).sum().cpu()
        return [lprob1, lprob2, lprob3]


class TransDs(Dataset):
    def __init__(self, fname, flores='/work/dlhlp2022/HW5/flores101_dataset/dev'):
        self.premise = list()
        self.hypothesis = list()
        self.lang = list()
        self.floresCode = {'en': 'eng', 'zh': 'zho_simpl', 'ru': 'rus', 'vi': 'vie',
                           'ur': 'urd', 'hi': 'hin', 'fr': 'fra', 'bg': 'bul'}
        self.fullang = ['English', 'French', 'Russian', 'Chinese', 'Hindi', 'Urdu', 'Bulgarian', 'Vietnamese']
        self.codes = ['en', 'fr', 'ru', 'zh', 'hi', 'ur', 'bg', 'vi']
        self.code_to_lang = dict(zip((self.codes), self.fullang))
        self.flore = {}
        self.xnlip = {}
        self.xnlih = {}
        for e in self.floresCode.keys():
            self.flore[e] = list()
            self.xnlip[e] = list()
            self.xnlih[e] = list()
            fl = os.path.join(flores, f'{self.floresCode[e]}.dev')
            for i, line in enumerate(open(fl).readlines()):
                self.flore[e].append(line)
                
            self.flore[e] = np.array(self.flore[e])
        for i, line in enumerate(open(fname).readlines()):
            if i == 0:
                continue
            line = line.split('\t')
            if line[0] in self.floresCode.keys():
                self.xnlip[line[0]].append(line[6])
                self.xnlih[line[0]].append(line[7])
                
        for e in self.xnlip.keys():
            self.xnlip[e] = np.array(self.xnlip[e])
            self.xnlih[e] = np.array(self.xnlih[e])
            
        # print('86', self.xnlip, self.xnlih, self.flore)
    def sample(self, src, tgt, sample_size=20):
        # print('88', len(self.flore[src]), len(self.xnlip[src]), len(self.xnlih[src]), src)
        fidx = np.random.choice(len(self.flore[src]), replace=False, size=(sample_size // 2))
        xpidx = np.random.choice(len(self.xnlip[src]), replace=False, size=(sample_size // 4))
        xhidx = np.random.choice(len(self.xnlih[src]), replace=False, size=(sample_size // 4))
        tshot = ''
        for s, t in zip(self.flore[src][fidx], self.flore[tgt][fidx]):
            tshot += (s + ' => ' + t + '\n')
        for s, t in zip(self.xnlip[src][xpidx], self.xnlip[tgt][xpidx]):
            tshot += (s + ' => ' + t + '\n')
        for s, t in zip(self.xnlih[src][xhidx], self.xnlih[tgt][xhidx]):
            tshot += (s + ' => ' + t + '\n')
            
        return tshot
            
class XNLIDs(Dataset):
    def __init__(self, fname):
        self.lang = list()
        self.label = list()
        self.premise = list()
        self.hypothesis = list()
        for i, line in enumerate(open(fname).readlines()):
            if i == 0:
                continue
            line = line.split('\t')
            self.lang.append(line[0])
            self.label.append(line[1])
            self.premise.append(line[6])
            self.hypothesis.append(line[7])
            
        self.lang = np.array(self.lang)
        self.label = np.array(self.label)
        self.premise = np.array(self.premise)
        self.hypothesis = np.array(self.hypothesis)
        self.label2prompt = {
            'R':'',
            "entailment": " , right? Yes, ", 
            "contradiction": " , right? No, ", 
            "neutral": " , right? Also, "
        }
        self.eprompts = {'R': "Is that correct?", 'entailment': "Yes, that is true", 
                         'contradiction': "No, that is not true", 'neutral':"Oh, besides"}

    def langcode(self, lang='en'):
        idxs = np.where(self.lang==lang)
        self.codep = self.premise[idxs]
        self.codeh = self.hypothesis[idxs]
        self.codelab = self.label[idxs]
    
    def getshot(self, shot=12):
        exs = list()
        for e in self.label2prompt.keys():
            if e == 'R':
                continue
            idxs = np.where(self.codelab == e)[0]
            # print('142', idxs, idxs.shape) 
            assert np.all(self.codelab[idxs]==e)
            idxs = np.random.choice(idxs, replace=False, size=shot // 3)
            
            exs.extend([bind(s, t, self.label2prompt['R'] + self.label2prompt[e]) for s, t in zip(self.codep[idxs], self.codeh[idxs])])
        
        exs = np.array(exs)
        np.random.shuffle(exs)
        return exs
    
    def __getitem__(self, index):
        return self.codep[index], self.codeh[index], self.codelab[index]
    
    def __len__(self):
        return len(self.codep)
# to 




# load xnli


def test_xnli(args):
    
    infer_dataset = XNLIDs('/work/dlhlp2022/HW5/XNLI-1.0/xnli.test.tsv')
    val_dataset = XNLIDs('/work/dlhlp2022/HW5/XNLI-1.0/xnli.dev.tsv')
    trans_ds = TransDs(fname='/work/dlhlp2022/HW5/XNLI-1.0/xnli.dev.tsv')
    # log_file = open('log', 'a')
    langs = ['en', 'fr', 'ru', 'zh', 'hi', 'ur', 'bg', 'vi']
    shot = [0, 12]
    model = Model()
    with open('./log_xglm_12', 'a+') as log_file:
        for lang in langs[:]:
            val_dataset.langcode(lang)
            infer_dataset.langcode(lang)
            task = 'Translate from English to {}: \n'.format(trans_ds.code_to_lang[lang])
            transp = task + trans_ds.sample('en', lang)
            max_len_b = 0
            for k in val_dataset.label2prompt.keys():
                hint = transp[:]
                hint += (val_dataset.eprompts[k] + ' =>')
                print(hint)
                max_len_b = len(val_dataset.eprompts[k].split()) * 1.2 + 100
                res = model.lm.translate(hint, beam=1, max_len_a=1.0,  max_len_b=max_len_b, replace_newlines_with_eos=True)
                val_dataset.label2prompt[k] = res.split('=>')[-1]
            # Start prediction
            
            print('183', val_dataset.label2prompt)
            for n in shot:
                torch.cuda.empty_cache()
                if n == 12 and lang != 'en':
                    # continue
                    pass
                print(f"=*=*=*=*= Running {n}-shot on {lang} =*=*=*=*=")
                acc = 0.0
                for i, e in enumerate(tqdm(infer_dataset)):
                    prefix = ""
                    samps = val_dataset.getshot(shot=n)
                    if not i:
                        print(samps)
                    for s in samps:
                        prefix += (s + '\n')
                    predict = model.XNLI_eval(prefix + e[0], e[1], val_dataset.label2prompt)
                    if predict == e[2]:
                        acc += 1.0
                avg_acc = acc/float(len(infer_dataset))
                print(f'Accuracy of {n}-shot on {lang}: {avg_acc}')
                print(f'Accuracy of {n}-shot on {lang}: {avg_acc}', file=log_file)
                print("=*=*=*=*=         End          =*=*=*=*=")


if __name__ == '__main__':
    args = parse_args()
    test_xnli(args)

