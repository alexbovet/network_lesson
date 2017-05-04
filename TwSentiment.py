# -*- coding: utf-8 -*-
"""
Created on May 2nd 2017

@author: Alexandre Bovet <alexandre.bovet@gmail.com>

Sentiment Analysis of tweets
"""


import collections

from nltk import ngrams
from itertools import chain
import numpy as np
from string import punctuation
from nltk.tokenize.casual import TweetTokenizer, _replace_html_entities, remove_handles, \
                                reduce_lengthening, HANG_RE, WORD_RE, EMOTICON_RE
import re


def bag_of_words(words):
    return dict([(word, True) for word in words])

def bag_of_words_and_bigrams(words):
    
    bigrams = ngrams(words, 2)
    
    return bag_of_words(chain(words, bigrams))    
       

#==============================================================================
# Custom Tokenizer for tweets
#==============================================================================



def normalize_mentions(text):
    """
    Replace Twitter username handles with '@USER'.
    """
    pattern = re.compile(r"(^|(?<=[^\w.-]))@[A-Za-z_]+\w+")
    return pattern.sub('@USER', text)


def normalize_urls(text):
    """
    Replace urls with 'URL'.
    """  
    pattern = re.compile(r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")
    # first shorten consecutive punctuation to 3 
    # to avoid the pattern to hang in exponential loop in extreme cases.    
    text = HANG_RE.sub(r'\1\1\1', text)

    return pattern.sub('URL', text)
    

def _lowerize(word, keep_all_upper=False):
    if EMOTICON_RE.search(word):
        return word
    elif word.isupper() and keep_all_upper:
        return word
    elif word == 'URL':
        return word
    elif word == '@USER':
        return word
    else:
        return word.lower()

class CustomTweetTokenizer(TweetTokenizer):
    """ Custom tweet tokenizer based on NLTK TweetTokenizer"""
    
    def __init__(self, preserve_case=False, reduce_len=True, strip_handles=False, 
                 normalize_usernames=True, normalize_urls=True, keep_allupper=True):
        
        TweetTokenizer.__init__(self, preserve_case=preserve_case, reduce_len=reduce_len, 
                                strip_handles=strip_handles)
                                
        self.keep_allupper = keep_allupper
        self.normalize_urls = normalize_urls
        self.normalize_usernames = normalize_usernames
        
        if normalize_usernames:
            self.strip_handles = False
        
        if self.preserve_case:
            self.keep_allupper = True
        
        
    def tokenize(self, text):
        """
        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings;
        
        Normalizes URLs, usernames and word lengthening depending of the
        attributes of the instance.
        
        """
        # Fix HTML character entities:
        text = _replace_html_entities(text)
        # Remove or replace username handles
        if self.strip_handles:
            text = remove_handles(text)
        elif self.normalize_usernames:
            text = normalize_mentions(text)
        
        if self.normalize_urls:
            # Shorten problematic sequences of characters
            text = normalize_urls(text)
        
        # Normalize word lengthening
        if self.reduce_len:
            text = HANG_RE.sub(r'\1\1\1', text)
            text = reduce_lengthening(text)
        
        # Tokenize:
        safe_text = HANG_RE.sub(r'\1\1\1', text)
        words = WORD_RE.findall(safe_text)
        
        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        # lower words but keep words that are all upper cases                              
        if not self.preserve_case:
            words = [_lowerize(w, self.keep_allupper) for w in words]
            
            
        return words
        



#==============================================================================
# Emoticon classification
#==============================================================================
    
POS_EMOTICONS = [":D", ":-D", ":-)", ":=)", "=)", "XD", "=D", "=]", ":]", ":<)",
                 ":>)", "=}", ":)",":}", ":o)","8D","8-)",
                 ":]", ":-}", ":-]",":-.)","^_^", "^-^"]  

NEG_EMOTICONS = [":(", ":-(", ":'(", "=(", "={", 
                ":-{", ":-{", ":-(", ":'{", "=[", ":["]
                
POS_EMOJIS_RE = re.compile(u'['
                         u'\U0001F600-\U0001F606'
                         u'\U0001F60A-\U0001F60E'
                         u'\U0001F638-\U0001F63B'
                         u'\U0001F642'
                         u'\U0000263A-\U0000263B]+', 
                         re.UNICODE)

NEG_EMOJIS_RE = re.compile(u'['
                        u'\U0001F61E-\U0001F622'
                        u'\U0001F63E-\U0001F63F'
                        u'\U0001F641'
                        u'\U00002639]+', 
                        re.UNICODE)
                        
def classifyEmoticons(text):
    
    # find all emoticons
    emoticons = EMOTICON_RE.findall(text)
    
    pos = any([emo in POS_EMOTICONS for emo in emoticons]) or bool(POS_EMOJIS_RE.search(text))
    neg = any([emo in NEG_EMOTICONS for emo in emoticons]) or bool(NEG_EMOJIS_RE.search(text))

    if pos and neg:
        return 'N/A'
    elif pos and not neg:
        return 'pos'
    elif neg and not pos:
        return 'neg'
    elif not pos and not neg:
        return None




class TweetClassifier(object):
    
    def __init__(self, classifier,
                 tokenizer=CustomTweetTokenizer(preserve_case=False,
                                 reduce_len=True, 
                                 strip_handles=False,
                                 normalize_usernames=False, 
                                 normalize_urls=False, 
                                 keep_allupper=False),
                  feature_extractor=bag_of_words_and_bigrams,
                  label_inv_mapper={0 : 'neg' , 1 : 'pos'},
                  polarity_threshold=0.5):
        
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.label_inv_mapper = label_inv_mapper
        self.polarity_threshold = polarity_threshold
        self.labels = [self.label_inv_mapper[c] for c in  self.classifier.classes_]
    
    def classify_text(self, text, return_pred_labels=True):
                          
    
        if isinstance(text, str):
            #single text
    
            tokens = self.tokenizer.tokenize(text)
                
            features = self.feature_extractor(tokens)
            
            proba = self.classifier.predict_proba(features)
            
            proba = proba.flatten()
            
            if return_pred_labels:
                if np.max(proba) > self.polarity_threshold:
                    
                    predicted_label = self.labels[np.argmax(proba)]
                
                else:
                    
                    predicted_label = 'N/A'
                
             
        elif isinstance(text, list):
            # list of multiple texts
    
            tokens = map(self.tokenizer.tokenize, text)
            features = map(self.feature_extractor, tokens)
            
            proba = self.classifier.predict_proba(features)
            
            if return_pred_labels:
                len_labels = max(len(l) for l in self.labels)
                
                predicted_label = np.zeros(len(text), dtype='<U' + str(len_labels))
                predicted_label[:] = 'N/A'
                
                mask = np.max(proba,axis=1) > self.polarity_threshold
                
                predicted_label[mask] = [self.labels[i] for i in np.argmax(proba[mask], axis=1)]
                            
    
        if return_pred_labels:                            
            return predicted_label, proba
        else:
            return proba
