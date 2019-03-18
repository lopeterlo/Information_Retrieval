import requests
import re
import sys
import math
from scipy import spatial
import pandas as pd
from nltk.corpus import stopwords

# for Porter's stemmer algorithm
class PorterStemmer:

    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[k0],
        b[k0+1] ... ending at b[k]. In fact k0 = 0 in this demo program. k is
        readjusted downwards as the stemming progresses. Zero termination is
        not in fact used in the algorithm.

        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.
        """

        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0   # j is a general offset into the string

    def cons(self, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        """m() measures the number of consonant sequences between k0 and j.
        if c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        """vowelinstem() is TRUE <=> k0,...j contains a vowel"""
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        """doublec(j) is TRUE <=> j,(j-1) contain a double consonant."""
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j-1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        """cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. this is used when trying to
        restore an e at the end of a short  e.g.

           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        """ends(s) is TRUE <=> k0,...k ends with the string s."""
        length = len(s)
        if s[length - 1] != self.b[self.k]: # tiny speed-up
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k-length+1:self.k+1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        """setto(s) sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        """r(s) is used further down."""
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):   self.setto("ate")
            elif self.ends("bl"): self.setto("ble")
            elif self.ends("iz"): self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):   self.r("ate")
            elif self.ends("tional"):  self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):      self.r("ence")
            elif self.ends("anci"):    self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):       self.r("ble") # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):    self.r("al")
            elif self.ends("entli"):   self.r("ent")
            elif self.ends("eli"):     self.r("e")
            elif self.ends("ousli"):   self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):   self.r("ize")
            elif self.ends("ation"):   self.r("ate")
            elif self.ends("ator"):    self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):     self.r("al")
            elif self.ends("iveness"): self.r("ive")
            elif self.ends("fulness"): self.r("ful")
            elif self.ends("ousness"): self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):     self.r("al")
            elif self.ends("iviti"):   self.r("ive")
            elif self.ends("biliti"):  self.r("ble")
        elif self.b[self.k - 1] == 'g': # --DEPARTURE--
            if self.ends("logi"):      self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() dels with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):     self.r("ic")
            elif self.ends("ative"):   self.r("")
            elif self.ends("alize"):   self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):      self.r("ic")
            elif self.ends("ful"):     self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"): pass
            else: return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"): pass
            elif self.ends("ence"): pass
            else: return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"): pass
            else: return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"): pass
            else: return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"): pass
            elif self.ends("ible"): pass
            else: return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"): pass
            elif self.ends("ement"): pass
            elif self.ends("ment"): pass
            elif self.ends("ent"): pass
            else: return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'): pass
            elif self.ends("ou"): pass
            # takes care of -ous
            else: return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"): pass
            else: return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"): pass
            elif self.ends("iti"): pass
            else: return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"): pass
            else: return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"): pass
            else: return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"): pass
            else: return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k -1

    def stem(self, p, i, j):
        """In stem(p,i,j), p is a char pointer, and the string to be stemmed
        is from p[i] to p[j] inclusive. Typically i is zero and j is the
        offset to the last character of a string, (p[j+1] == '\0'). The
        stemmer adjusts the characters p[i] ... p[j] and returns the new
        end-point of the string, k. Stemming never increases word length, so
        i <= k <= j. To turn the stemmer into a module, declare 'stem' as
        extern, and delete the remainder of this file.
        """
        # copy the parameters into statics
        self.b = p
        self.k = j
        self.k0 = i
        if self.k <= self.k0 + 1:
            return self.b # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k+1]


def main():
    stop_words = set(stopwords.words('english'))
    # res = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words')
    # for word in res.text.split('\n'):
    #     word = word.replace('\r','')
    #     stop_words.append(word.lower())
    p = PorterStemmer()
    word_list = []
    output = []
    _id = 0
    term_in_art = dict()
    training_data = dict()
    all_label_doc = set()
    with open ('training.txt','r') as f1:
        for i in f1.readlines():
            splited = i.split(' ',1)
            class_id =  splited[0]
            art_ids = splited[1].replace('\n','')
            temp = []
            for art in art_ids[:-1].split(' '):
                temp.append(int(art))
                all_label_doc.add(int(art))
            training_data[class_id] = temp
    
    for i in all_label_doc:
        try:
            with open ('IRTM/' + str(i) + '.txt', 'r') as f :
                temp = [] # record term id 
                data = re.split(' |\.|\'|\r|\n|\,|\?|\`|\(|\)|\-|\@|\"|\:|\_|\%|\#|\;|\/|\*|\$|\&|\!', f.read())
                for token in data :
                    token = token.lower()
                    if token is '' or len(token) < 2:
                        continue
                    if token in stop_words:
                        continue
                    if re.search(r'\d', token): # if token has number in it 
                        continue
                    stemmed_word = p.stem(token, 0,len(token)-1)
                    if stemmed_word not in word_list:
                        word_list.append(stemmed_word)
                        output.append({'term': stemmed_word, 'df': 1 , 'all-tf':[{'id': i, 'tf':1}], 'id': _id, 'arts': set([i])})
                        temp.append(_id)
                        _id += 1
                    else:
                        index = word_list.index(stemmed_word)
                        flag = 0
                        for info in output[index]['all-tf']:
                            if info['id'] == i:
                                info['tf'] +=1
                                flag = 1
                                break
                        if not flag:
                            output[index]['all-tf'].append({'id': i, 'tf':1})
                            output[index]['df'] += 1
                        output[index]['arts'].add(i)
                        temp.append(output[index]['id'])
                            
                term_in_art[i] = list(set(temp))
            print('finished' + ' '+str(i) + ' ' + 'document' )
        except Exception as e:
            #print(e)
            print('finish')
            break
    
    num_doc = len(all_label_doc)
    features_id = dict()
    condprob = dict()
    prior = dict()
    all_tf = dict()
    all_features = []
    
    for key, val in training_data.items():
        other_label_doc = all_label_doc.difference(set(val))
        class_id = key 
        class_doc_num = len(val)
        prior[key] = class_doc_num / num_doc
        features_id[class_id]= select_feature(val, class_id, 500, output, term_in_art, other_label_doc, 'mix') # 500 top value in class docs
        all_features += features_id[class_id]
    
    all_features = set(all_features)

    for key, val in training_data.items():
        all_term_tf = 0
        class_id = key 
        count = 0
        for t in all_features:
            tf_in_class = 0
            for info in output[t]['all-tf']:
                if info['id'] in val:
                    tf_in_class += info['tf']
            all_tf[f'{output[t]["term"]}_{class_id}'] = tf_in_class
            all_term_tf += tf_in_class
        for t in all_features:
            condprob[f'{output[t]["term"]}_{class_id}'] = (all_tf[f'{output[t]["term"]}_{class_id}'] + 1) / (all_term_tf + len(features_id[class_id]))
        print(f'class {class_id} finished')
    text_file = 1 

    label_list = []
    id_list = []
    while True:
        if text_file in all_label_doc:
            text_file += 1  
            continue 
        try:
            all_words = list()
            with open ('IRTM/' + str(text_file) + '.txt', 'r') as f :
                id_list.append(text_file)
                data = re.split(' |\.|\'|\r|\n|\,|\?|\`|\(|\)|\-|\@|\"|\:|\_|\%|\#|\;|\/|\*|\$|\&|\!', f.read())
                for token in data :
                    token = token.lower()
                    if token is '' or len(token) < 2:
                        continue
                    if token in stop_words:
                        continue
                    if re.search(r'\d', token): # if token has number in it 
                        continue
                    stemmed_word = p.stem(token, 0,len(token)-1)
                    all_words.append(stemmed_word)
            score = dict()
            for class_id in training_data.keys():
                score[class_id] = math.log(prior[class_id])
                for term in all_words:
                    if condprob.get(f'{term}_{class_id}') != None:
                        score[class_id] += math.log(condprob[f'{term}_{class_id}'])
            max_val = -100000
            label = -1
            for key, value in score.items():
                if value > max_val:
                    max_val = value
                    label = key
            label_list.append(label)
            print(f'finished doc {text_file}')
            text_file += 1
        except Exception as e:
            #print(e)
            print('finish')
            break
    df = pd.DataFrame({'id': id_list,'Value':label_list})
    df = df.astype(int)
    df.to_csv('result.csv', index= False)


def select_feature(doc_ids, class_id, count, output, term_in_art, other_label_doc, method):
    if method == 'likelyhood':
        val_list = dict()
        for doc in doc_ids:
            for term in term_in_art[doc]:
                n11 = len(output[term]['arts'].intersection(doc_ids))
                n01 = len(output[term]['arts'].intersection(other_label_doc))
                n10 = len(doc_ids) - n11
                n00 = len(doc_ids)+ len(other_label_doc) - n01 - n11- n10
                N = n11 + n00 + n01 + n10
                numerator = math.pow((n11+n01)/N, n11 + n01)  * math.pow(1-(n11+n01)/N, n10+n00) 
                denominator = math.pow( n11 / (n11 + n10), n11 )  * math.pow(1-(n11 / (n11 + n10)), n10) * \
                            math.pow( n01 / (n01 + n00), n01 )  * math.pow(1-(n01 / (n01 + n00)), n00) 
                val =  -2 * math.log(numerator/ denominator, 10) 
                val_list[term] = val
        ret = sorted(val_list.items(), key=lambda d: d[1],reverse=True)
        ans_id = []
        ans_term = []
        for i in ret[:38]:
            #print(output[i[0]]['term'], i[1])
            ans_id.append(i[0])
        return ans_id
    elif method == 'chi': # chi
        val_list = dict()
        test = dict()
        for doc in doc_ids:
            for term in term_in_art[doc]:
                n11 = len(output[term]['arts'].intersection(doc_ids)) 
                n01 = len(output[term]['arts'].intersection(other_label_doc))
                if n11 ==0:
                    continue
                n10 = len(doc_ids) - n11
                n00 = len(doc_ids)+ len(other_label_doc) - n01 - n11- n10
                N = n11 + n00 + n01 + n10
                e11 = N * (n11+n01)/N * (n11+n10)/N
                e01 = N * (n01+n00)/N * (n11+n01)/N
                e10 = N * (n11+n10)/N * (n00+n10)/N
                e00 = N * (n00+n01)/N * (n00+n10)/N
                val =  math.pow((n11-e11), 2) / e11 +  math.pow((n10-e10), 2) / e10 +  math.pow((n01-e01), 2) / e01 +  math.pow((n00-e00), 2) / e00
                val_list[term] = val
        ret = sorted(val_list.items(), key=lambda d: d[1],reverse=True)
        ans_id = []
        ans_term = []
        flag = 0
        for i in ret[:38]:
            ans_id.append(i[0])
        return ans_id
    elif method == 'MI':
        val_list = dict()
        for doc in doc_ids:
            for term in term_in_art[doc]:
                n11 = len(output[term]['arts'].intersection(doc_ids)) 
                n01 = len(output[term]['arts'].intersection(other_label_doc))
                n10 = len(doc_ids) - n11
                n00 = len(doc_ids)+ len(other_label_doc) - n01 - n11- n10
                N = n11 + n00 + n01 + n10
                n1_ = n11 + n10
                n_1 = n01 + n11
                n0_ = n00 + n01
                n_0 = n00 + n10
                first = (n11/N)* math.log((N*n11)/(n1_ +n_1),2) if n11 else 0
                second = (n01/N)* math.log((N*n01)/(n0_ +n_1),2) if n01 else 0
                third = (n10/N)* math.log((N*n10)/(n1_ +n_0),2) if n10 else 0
                forth = (n00/N)* math.log((N*n00)/(n0_ +n_0),2) if n00 else 0
                val =  first + second + third + forth
                val_list[term] = val
        ret = sorted(val_list.items(), key=lambda d: d[1],reverse=True)
        ans_id = []
        ans_term = []
        for i in ret[:38]:
            ans_id.append(i[0])
        return ans_id
    elif method == 'mix':
        ans = set(select_feature(doc_ids, class_id, count, output, term_in_art, other_label_doc, 'chi'))
        ans = ans.intersection(set(select_feature(doc_ids, class_id, count, output, term_in_art, other_label_doc, 'MI'))) 
        ans = ans.intersection(set(select_feature(doc_ids, class_id, count, output, term_in_art, other_label_doc, 'likelyhood')))    
        return ans
    else:
        return None
def asending(elem):
    return elem['term']


if __name__ == '__main__':
    main()

