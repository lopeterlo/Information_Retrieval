import requests
import re
import sys
import math
from scipy import spatial
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


def swap_max(a,b):
    if a > b:
        temp = a 
        a = b 
        b = temp
    return b, a

def heapify(data, root, length, mapping):
    left = root * 2 +1
    right = root * 2 + 2
    max_node = -1
    if (left < length and ( list(data[left].values())[0] > list(data[root].values())[0]) ):
        max_node = left
    else:
        max_node = root
    if (right < length and (list(data[right].values())[0] > list(data[max_node].values())[0]) ):
        max_node = right
    if (max_node != root):
        # print(data[root], data[max_node])
        
        a = 0
        for key, val in data[root].items():
            a = key
        b = 0
        for key, val in data[max_node].items():
            b = key
        mapping[a] = max_node
        mapping[b] = root
        temp = data[root]
        data[root] = data[max_node]
        data[max_node] = temp


DOC_NUM = 1095

def swap(a,b):
    if a > b:
        temp = a 
        a = b 
        b = temp
    return a, b


def main():
    stop_words = set(stopwords.words('english'))
    p = PorterStemmer()
    count = 1
    fileName = str(count) + '.txt'
    word_list = []
    output = []
    _id = 0
    term_in_art = []
    while True:
        try:
            with open ('IRTM/' + fileName, 'r') as f :
                df_num = False
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
                        output.append({'term': stemmed_word, 'df': 1 , 'all-tf':[{'id':count, 'tf':1}], 'id': _id})
                        temp.append(_id)
                        _id += 1
                        df_num = True
                    else:
                        index = word_list.index(stemmed_word)
                        flag = 0
                        for data in output[index]['all-tf']:
                            if data['id'] == count:
                                data['tf'] +=1
                                flag = 1
                            break
                        if not flag:
                            output[index]['all-tf'].append({'id': count, 'tf':1})
                        temp.append(output[index]['id'])
                        # if output[index]['word'] != stemmed_word:
                        #     print('error')
                        #     break
                        if not df_num:
                            output[index]['df'] += 1
                            df_num = True
                term_in_art.append(list(set(temp)))
            print('finished' + ' '+str(count) + ' ' + 'document' )
            count += 1
            fileName = str(count) + '.txt'
        except Exception as e:
            #print(e)
            print('finish')
            break

    #2 
    num_doc = 1095
    count = 1
    fileName = str(count) + '.txt'
    all_doc = [[] for i in range(num_doc)] 
    for art in term_in_art :
        doc = [0 for i in range(len(output))]
        try:  
            for term_id in art:
                tf = None
                df = output[term_id]['df']
                for data in output[term_id]['all-tf']:
                    if data['id'] == count:
                        tf = data['tf']
                        break
                tf_idf = math.log(num_doc / df, 10) * tf
                doc [term_id] = tf_idf
            all_doc[count-1] = doc 
            print('finished transform to word vector: ' + ' '+str(count) + ' ' + 'document' )
            count += 1
            fileName = str(count) + '.txt'
        except Exception as e:
            print(e)
            print('finish')
            break
    
    # calculate similarity
    avail_clus = [1 for i in range(0, DOC_NUM)]
    priority = dict()
    clusters = dict()
    # with open ('temp_sim', 'w') as f:
    # for i in range(0, DOC_NUM):
    #     temp_dict = dict()
    #     for j in range(i+1, DOC_NUM):
    #         if i != j:
    #             sim_info = dict()
    #             sim_info = {j: cos_similarity(all_doc[i], all_doc[j])}
    #             temp_dict.update(sim_info)
    #     #         f.write(str(cos_similarity(all_doc[i], all_doc[j])))
    #     #         f.write(' ')
    #     # f.write('\n')        
    #     clusters[i] = temp_dict
    #     max_list =  sorted(temp_dict.items(), key=lambda d: d[1],reverse=True)
    #     temp = list()
    #     for key, val in max_list:
    #         temp.append(key)
    #     print(i)
    #     priority[i] = temp
    all_prio = list()
    mapping = dict()

    with open('temp_sim', 'r') as f:
        front_ind = 0
        for i in f.readlines():
            temp = dict()
            index = front_ind+1
            for k in i.split(' '):
                try:
                    if k != '\n':
                        temp[index] = float(k)
                except :
                    continue
                index += 1
            clusters[front_ind] = temp
            max_list =  sorted(temp.items(), key=lambda d: d[1],reverse=True)
            temp_data = list()
            for key, val in max_list:
                temp_data.append(key)
            priority[front_ind] = temp_data
            front_ind += 1   

    
    merge_list = dict()
    central = dict()
    doc_len = dict()


    for i in range(0, DOC_NUM):
        merge_list[i] = [i]
        central[i] = all_doc[i]
        doc_len[i] = 1
        
    
    while(True):
        ### heap
        ####
        print(len(merge_list))
        i, j = get_highest_sim(priority, clusters, avail_clus)
        print(i, j)
        #if centroid method
        # doc_com = doc_len[i] + doc_len[j]
        # temp = merge( multiply(central[i], doc_len[i]) , multiply(central[j], doc_len[j]) )
        # central[i] = multiply( temp, 1/doc_com )
        # doc_len[i] = doc_com
        ###########
        for val in merge_list[j]:
            merge_list[i].append(val)

        merge_list.pop(j)
        avail_clus[j] = 0
        priority[i] = []
        priority[j] = []
        for num in range(0, DOC_NUM):
            if avail_clus[num] ==1  and num != i:
                if num < j:
                    priority[num].remove(j)
                if num < i:    
                    priority[num].remove(i)
                    clusters[num][i] = complete_link(num, i, j, clusters)
                    # clusters[num][i] = centroid_cluster(num, i, central)
                    index = insert_new(num, priority[num], clusters,  clusters[num][i])
                    priority[num].insert(index, i)
                else:
                    # clusters[i][num] = centroid_cluster(num, i, central)
                    clusters[i][num] = complete_link(num, i, j, clusters)
                    index = insert_new(i, priority[i], clusters,  clusters[i][num])
                    priority[i].insert(index, num)
                


        if len(merge_list) == 20 or len(merge_list) == 13 or len(merge_list) == 8:
            name = 'result_' + str(len(merge_list))+ '.txt'
            with open (name, 'w') as f:
                for key, val in merge_list.items():
                    val.sort()
                    for _id in val:
                        f.write(str(_id+1))
                        f.write('\n')
                    f.write('\n')
                    
            if len(merge_list)==8:
                break

def insert_new(i, pri, clu, val):
    index = 0
    # print(pri)
    # print(clu)
    for k in range(0, len(pri)):
        if clu[i].get(pri[k]) :
            if clu[i][pri[k]] <= val:
                index = k
                break
    return index
        

def single_link(i, j ,k, clusters):
    k1, k2 = swap(i, j)
    k3, k4 = swap(i, k)
    return max(clusters[k1][k2], clusters[k3][k4])

def complete_link(i, j ,k, clusters):
    k1, k2 = swap(i, j)
    k3, k4 = swap(i, k)
    return min(clusters[k1][k2], clusters[k3][k4])


def centroid_cluster(i, j ,central):
    return cos_similarity(central[i], central[j])


def multiply(cen_list, y):
    for i in range(len(cen_list)):
        cen_list[i] = cen_list[i] * y
    return cen_list  


def merge(a, b):
    for i in range(len(a)):
        a[i] += b[i]
    return a

def get_highest_sim(priority, clusters, avail):
    max_val = -10000
    i1 = -1
    i2 = -2
    # DOC_NUM -1 
    for i in range(0, DOC_NUM-1):
        if avail[i] == 0:
            continue
        i3 = i
        try:
            i4 = priority[i][0]
        except:
            continue
        i3, i4 = swap(i3, i4)
        temp = clusters[i3][i4]
        if temp > max_val:
            max_val = temp
            i1 = i3
            i2 = i4
    return i1, i2


def cos_similarity(doc1, doc2):
    result = 1 - spatial.distance.cosine(doc1, doc2)
    # print(result)
    return result



if __name__ == '__main__':
    main()

    