import random
import numpy as np

from tqdm import tqdm
from fairseq.models.transformer_lm import TransformerLanguageModel
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

class Model():
    def __init__(self, model_dir='/work/dlhlp2022/HW5/Assignment-cross-lingual-transfer/7.5B'):
        self.model_dir = model_dir
        self.lm = TransformerLanguageModel.from_pretrained(self.model_dir, bpe='sentencepiece')
        self.lm = self.lm.eval()
        self.lm = self.lm.half()
        self.lm = self.lm.cuda()
        self.pred_label = np.array(['entailment', 'contradiction', 'neutral'])
        self.languages = ['English', 'French', 'Russian', 'Chinese', 'Hindi', 'Urdu', 'Bulgarian', 'Vietnamese']
        self.codes = ['en', 'fr', 'ru', 'zh', 'hi', 'ur', 'bg', 'vi']
        self.lang_prompt = {'ur': ['، ٹھیک ہے؟ جی ہاں،', '، ٹھیک ہے؟ نہیں،', '، ٹھیک ہے؟ اس کے علاوہ،'], 
                            'hi': [', सही? हां,', ', सही? नहीं,', ', सही? भी,'], 
                            'vi': [', đúng? Đúng,', ', đúng? Không,', ', đúng? Cũng thế,'], 
                            'bg': [', нали? да,', ', нали? Не,', ', нали? Също,'], 
                            'zh': ['， 對？是的，', '， 對？不，', '， 對？還，'], 
                            'ru': [', Правильно? Да,', ', Правильно? Нет,', ', Правильно? Также,'],
                            'fr': [', à droite? Oui,', ', à droite? Non,', ', à droite? Aussi,']}
        self.code_to_lang = dict(zip((self.codes), self.languages))
        self.eprompts = ["Is that true?", "Yes, that is true", "No, that is not true", "Oh, besides"]
        self.op = [" , right? Yes, ", " , right? No, ", " , right? Also, "]
        self.lang_prompt['en'] = self.eprompts
        self.prompts = self.eprompts
    def get_logprobs(self, prompt):
        import re
        prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines, which indicate separate documents
        return self.lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']
    # to 
    def XNLI_eval(self, premise, hypothesis):
        p1 = " , "
        p2 = ", "
        
        lprob1 = self.get_logprobs(premise + p1 + self.prompts[0] + self.prompts[1] + p2 + hypothesis).sum().cpu()
        lprob2 = self.get_logprobs(premise + p1 + self.prompts[0] + self.prompts[2] + p2 + hypothesis).sum().cpu()
        lprob3 = self.get_logprobs(premise + p1 + self.prompts[0] + self.prompts[3] + p2 + hypothesis).sum().cpu()
        '''
        lprob1 = self.get_logprobs(premise + self.prompts[0] + hypothesis).sum().cpu()
        lprob2 = self.get_logprobs(premise + self.prompts[1] + hypothesis).sum().cpu()
        lprob3 = self.get_logprobs(premise + self.prompts[2] + hypothesis).sum().cpu()
        '''
        return self.pred_label[np.argmax([lprob1, lprob2, lprob3])]

    def val_all(self, premise, hypothesis):
        lprob1 = self.get_logprobs(premise + " , right? Yes, " + hypothesis).sum().cpu()
        lprob2 = self.get_logprobs(premise + " , right? No, " + hypothesis).sum().cpu()
        lprob3 = self.get_logprobs(premise + " , right? Also, " + hypothesis).sum().cpu()
        return [lprob1, lprob2, lprob3]

    def trans_prompt(self, code, eng, other, shot=32):
        # self.prompts = self.lang_prompt[code]
        # return
        idxs = np.random.choice(len(eng), replace=False, size=shot)
        eng = eng[idxs]
        other = other[idxs]
        task = 'Translate from English to {}: \n'.format(self.code_to_lang[code])
        hint = task[:]
        for e, o in zip(eng, other):
            hint += (f'{e} => {o}\n')
        min_len_bs = list()
        exps = list()
        max_len_b = 0
        for prompt in self.eprompts:
            example = hint + prompt + '=>'
            print(example)
            max_len_b = len(prompt.split()) * 1.2 + 100
            min_len_bs.append(max_len_b)
            exps.append(example)

        min_len_b = int(np.max(min_len_bs))
        
        pred_pmpts = self.lm.translate(exps, beam=1, max_len_a=1.0,  max_len_b=min_len_b, replace_newlines_with_eos=True)
        
        self.prompts = [e.split('=>')[-1] for e in pred_pmpts]
        print(self.prompts)

class Dataset():
    def __init__(self, data_dir='/work/dlhlp2022/HW5/XNLI-1.0/', file_name='xnli.test.tsv'):
        self.lang = []
        self.label = []
        self.premise = []
        self.hypothesis = []
        self.data_path = data_dir + file_name
        for i, line in enumerate(open(self.data_path).readlines()):
            if i == 0:
                continue
            line = line.split('\t')
            self.lang.append(line[0])
            self.label.append(line[1])
            self.premise.append(line[6])
            self.hypothesis.append(line[7])
    
    def get_lang_data(self, select_lang='en'):
        lang = np.array(self.lang)
        label = np.array(self.label)
        premise = np.array(self.premise)
        hypothesis = np.array(self.hypothesis)

        idx = np.where(lang == select_lang)
        label = label[idx]
        premise = premise[idx]
        hypothesis = hypothesis[idx]
        
        return label, premise, hypothesis

def random_sample(arr, sample_size):
    result = np.random.choice(arr, replace=False, size=(sample_size))
    result = [i.item() for i in result]
    return result
    
    
def test_n_shot(n_shot, model, lang, label, premise, hypothesis, dev_label, dev_premise, dev_hypothesis, log_file, args):
    acc = 0.0
    if n_shot == 0:
        for i in tqdm(range(len(label))):
            predict = model.XNLI_eval(premise[i], hypothesis[i])
            if predict == label[i]:
                acc += 1.0
    else:
        label2prompt = {
            "entailment": " , right? Yes, ", 
            "contradiction": " , right? No, ", 
            "neutral": " , right? Also, "
        }
        # Balancing label distribution
        if n_shot == 1:
            idx = random_sample(np.where(dev_label == "entailment")[0], 1)
        elif n_shot == 2:
            idx = \
                random_sample(np.where(dev_label == "entailment")[0], 1) + \
                random_sample(np.where(dev_label == "contradiction")[0], 1)
        else:
            if n_shot % 3 == 1:
                idx = \
                    random_sample(np.where(dev_label == "entailment")[0], n_shot//3 + 1) + \
                    random_sample(np.where(dev_label == "contradiction")[0], n_shot//3) + \
                    random_sample(np.where(dev_label == "neutral")[0], n_shot//3)
            elif n_shot % 3 == 2:
                idx = \
                    random_sample(np.where(dev_label == "entailment")[0], n_shot//3 + 1) + \
                    random_sample(np.where(dev_label == "contradiction")[0], n_shot//3 + 1) + \
                    random_sample(np.where(dev_label == "neutral")[0], n_shot//3)
            else:
                idx = \
                    random_sample(np.where(dev_label == "entailment")[0], n_shot//3) + \
                    random_sample(np.where(dev_label == "contradiction")[0], n_shot//3) + \
                    random_sample(np.where(dev_label == "neutral")[0], n_shot//3)
        
        # Example for model to perform in-context learning
        prefix = ""
        for i in idx:
            prefix += (dev_premise[i] + label2prompt[dev_label[i]] + dev_hypothesis[i] + "\n")
        # Start prediction
        for i in tqdm(range(len(label))):
            predict = model.XNLI_eval(prefix + premise[i], hypothesis[i])
            if predict == label[i]:
                acc += 1.0

    avg_acc = acc/float(len(label))
    print(f'Accuracy of {n_shot}-shot on {lang}: {avg_acc}', file=log_file)
    print(f'Accuracy of {n_shot}-shot on {lang}: {avg_acc}')

def test_xnli(args):
    model = Model()
    infer_dataset = Dataset()
    dev_dataset = Dataset(file_name='xnli.dev.tsv')
    log_file = open('log', 'a')
    langs = ['en', 'fr', 'ru', 'zh', 'hi', 'ur', 'bg', 'vi']
    shot = [12]
    for lang in langs[args.lang_id: args.lang_id + 1]:
        label, premise, hypothesis = infer_dataset.get_lang_data(select_lang=lang)
        dev_label, dev_premise, dev_hypothesis = dev_dataset.get_lang_data(select_lang=lang)
        _, eng_premise, eng_hyp = dev_dataset.get_lang_data(select_lang='en')
        # From 0-shot to 12-shot
        model.trans_prompt(lang, eng_premise, dev_premise)
        for n in shot:
            print(f"=*=*=*=*= Running {n}-shot on {lang} =*=*=*=*=")
            test_n_shot(n, model, lang, label, premise, hypothesis, dev_label, dev_premise, dev_hypothesis, log_file, args)
            print("=*=*=*=*=         End          =*=*=*=*=")


if __name__ == '__main__':
    args = parse_args()
    test_xnli(args)

