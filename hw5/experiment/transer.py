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
    parser.add_argument(
        "--gpu_id",
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
        lprob1 = self.get_logprobs(premise + " , right? Yes, " + hypothesis).sum().cpu()
        lprob2 = self.get_logprobs(premise + " , right? No, " + hypothesis).sum().cpu()
        lprob3 = self.get_logprobs(premise + " , right? Also, " + hypothesis).sum().cpu()
        return self.pred_label[np.argmax([lprob1, lprob2, lprob3])]
        
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
        if sample_size == 0:
            return ''
        # print('88', len(self.flore[src]), len(self.xnlip[src]), len(self.xnlih[src]), src)
        fidx = np.random.choice(len(self.flore[src]), replace=False, size=(sample_size // 2))
        xpidx = np.random.choice(len(self.xnlip[src]), replace=False, size=(sample_size // 4))
        xhidx = np.random.choice(len(self.xnlih[src]), replace=False, size=(sample_size // 4))
        tshot = ''
        for s, t in zip(self.flore[src][fidx], self.flore[tgt][fidx]):
            s = s.strip('\n')
            t = t.strip('\n')
            tshot += (s + ' => ' + t + '\n')
        for s, t in zip(self.xnlip[src][xpidx], self.xnlip[tgt][xpidx]):
            s = s.strip('\n')
            t = t.strip('\n')
            tshot += (s + ' => ' + t + '\n')
        for s, t in zip(self.xnlih[src][xhidx], self.xnlih[tgt][xhidx]):
            s = s.strip('\n')
            t = t.strip('\n')
            tshot += (s + '=> ' + t + '\n')
            
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
    with open(f'./{langs[args.lang_id]}_to_en.tsv', 'w') as log_file:
        for lang in langs[args.lang_id:args.lang_id + 1]:
            val_dataset.langcode(lang)
            infer_dataset.langcode(lang)
            task = 'Translate from {} to English: \n'.format(trans_ds.code_to_lang[lang])
            transp = task + trans_ds.sample(lang, 'en', sample_size=0)
            print('183', val_dataset.label2prompt)
            torch.cuda.empty_cache()
            premise_examples = list()
            hypotheses_examples = list()
            min_len_b_premises = list()
            min_len_b_hypotheses = list()
            max_len_b = 0
            current_labels = infer_dataset.codelab
            for i, e in enumerate(tqdm(infer_dataset)):
                transp = task + trans_ds.sample(lang, 'en', sample_size=12)
                if len(premise_examples) > 3:
                    pass
                prefix = ""
                premise = e[0]
                hypothesis = e[1]
                # print('194', premise, hypothesis)
                example = transp + premise.strip('\n') + '<eos> => '
                if lang == 'zh':
                    max_len_b = len(premise) * 1.2 + 100
                else:
                    max_len_b = len(premise.split()) * 1.2 + 100

                premise_examples.append(example)
                min_len_b_premises.append(max_len_b)
                example = transp + hypothesis.strip('\n') + '<eos> => '

                if lang == 'zh':
                    max_len_b = len(hypothesis) * 1.2 + 100
                else:
                    max_len_b = len(hypothesis.split()) * 1.2 + 100

                hypotheses_examples.append(example)
                min_len_b_hypotheses.append(max_len_b)
            
            
            min_len_b = float(np.max(min_len_b_premises))
            # print('213', premise_examples, '214\n', hypotheses_examples, min_len_b, len(infer_dataset), len(current_labels))
            pred_premises = model.lm.translate(premise_examples, beam=1, max_len_a=1.0,  max_len_b=min_len_b, replace_newlines_with_eos=True)
            min_len_b = float(np.max(min_len_b_hypotheses))
            pred_hypotheses = model.lm.translate(hypotheses_examples, beam=1, max_len_a=1.0,  max_len_b=min_len_b, replace_newlines_with_eos=True)
            # print('218', pred_premises, pred_hypotheses)
            for pred_premise, pred_hypothesis, label in zip(pred_premises, pred_hypotheses, current_labels):
                pred_premise = pred_premise.split('=>')[-1]
                pred_hypothesis = pred_hypothesis.split('=>')[-1]
                log_file.write('{}\t{}\t{}\t{}\n'.format(lang, label, pred_premise, pred_hypothesis))
                    
                    
                


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(0)
    test_xnli(args)

