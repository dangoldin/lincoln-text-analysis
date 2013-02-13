# -*- coding: utf-8 -*-

import os
import re
import pickle
import json

from optparse import OptionParser

from bs4 import BeautifulSoup

from lxml.cssselect import CSSSelector
from lxml.html import fromstring

from pyquery import PyQuery
from lxml import etree

from dateutil.parser import parse

from nltk import ingrams
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from collections import defaultdict

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

# python analyze.py --d=html --o=out --p=data.pkl -l -a
# python analyze.py --d=html --o=out --p=data.pkl -a

RE_CLEAN_TEXT = re.compile(r'[\s+\n\t]')
RE_DATE = re.compile(r'\w+\s\d+,?\s\d+')

year_ranges = {
    '1832-1845' : [str(x) for x in range(1832,1846)],
    '1846-1853' : [str(x) for x in range(1846,1854)],
    '1854-1859' : [str(x) for x in range(1854,1860)],
    '1860' : ['1860'],
    '1861' : ['1861'],
    '1862' : ['1862'],
    '1863' : ['1863'],
    '1864' : ['1864'],
    '1865' : ['1865'],
}

year_to_range = {}
for year, year_range in year_ranges.items():
    for y in year_range:
        year_to_range[y] = year

def get_year_range(year):
    return year_to_range[year]

def year_quarter(year_mo):
    year, mo = year_mo.split('-')

    if mo in ('01','02','03'):
        q = 'Q1'
    elif mo in ('04','05','06'):
        q = 'Q2'
    elif mo in ('07','08','09'):
        q = 'Q3'
    elif mo in ('10','11','12'):
        q = 'Q4'
    return '%s-%s' % (year, q)

def clean_words(words):
    # return [w.lower() for w in words if not w.lower()
    #     in stopwords.words('english') and w.isalpha()]
    return [w.lower() for w in words if w.isalpha()]

def ngram_phrases(t, n):
    tokens = wordpunct_tokenize(t)
    tokens = clean_words(tokens)
    return [' '.join(n_grams) for n_grams in ingrams(tokens, n)]

# Can't one line this due to it being a generator
def count_words(t):
    cnt = 0
    for s in sent_tokenize(t):
        for w in word_tokenize(s):
            cnt += 1
    return cnt

# Can't one line this due to it being a generator
def count_sentences(t):
    cnt = 0
    for s in sent_tokenize(t):
        cnt += 1
    return cnt

def sentence_lengths(t):
    return [len(word_tokenize(s)) for s in sent_tokenize(t)]

def parsedate(date_str):
    m = RE_DATE.search(date_str)
    if m:
        try:
            return parse(m.group(0))
        except ValueError:
            return None
        except Exception, e:
            logger.error('Could not parse date %s: %s' % (date_str, str(e)))
    return None

def generate_dict_csv(cols, dist, outfile):
    keys = sorted(dist.keys())
    lines = [','.join(cols) + '\n']

    lines.extend('%s,%s\n' % (str(key),str(dist[key])) for key in keys)

    f = open(outfile, 'w')
    f.write(''.join(lines))
    f.close()

def generate_stream_js(year_freq_dist, outfile):
    top_phrases = FreqDist()
    all_years = sorted(year_freq_dist.keys())
    # Get the top phrases for each year
    for year in all_years:
        for phrase, cnt in year_freq_dist[year].items()[:20]:
            top_phrases.inc(phrase, cnt)

    json_data = []

    for i, phrase in enumerate(top_phrases.keys()[:20]):
        d = {
            'name' : phrase,
            'key' : i,
            'values' : [ { 'x': year, 'y' : year_freq_dist[year].get(phrase,0) } for year in all_years ]
        }
        json_data.append(d)

    f = open(outfile, 'w')
    f.write(json.dumps(json_data))
    f.close()

    # Get the total year counts
    year_counts = defaultdict(int)
    for year in all_years:
        year_counts[year] = sum(year_freq_dist[year].values())

    # Write the matrix
    matrix = [ ['phrase'] + all_years ]
    for phrase in top_phrases.keys()[:100]:
        row = [phrase]
        row.extend(str(year_freq_dist[year].get(phrase,0)/(1.0*year_counts[year])) for year in all_years)
        matrix.append(row)

    f = open(outfile.replace('.json','-matrix.csv'), 'w')
    for row in matrix:
        f.write(','.join(row) + '\n')
    f.close()

def generate_cloud_csv(year_freq_dist, outfile):
    all_years = sorted(year_freq_dist.keys())
    lines = ['year,phrase,cnt\n']

    lines.extend('%s,%s,%d\n' % (year,phrase,cnt) for year in all_years for phrase, cnt in year_freq_dist[year].items()[:50])
    f = open(outfile, 'w')
    f.write(''.join(lines))
    f.close()

def generate_sentence_length_csv(sentence_lengths, outfile):
    all_years = sorted(sentence_lengths.keys())
    lines = ['year,avg_length,num_sentences\n']

    for year in all_years:
        lengths = sentence_lengths[year]
        lines.append('%s,%f,%d\n' % (year, 1.0*sum(lengths)/len(lengths), len(lengths)))

    f = open(outfile,'w')
    f.write(''.join(lines))
    f.close()

def process_html_file(fi):
    f = open(fi,'r')
    d = PyQuery(f.read())
    f.close()

    year = None
    links = {}
    for a in d.items('.toc a'):
        if len(a.text()) == 4:
            year = a.text()
            links[year] = []
        else:
            if year:
                links[year].append( (a.attr('href'), a.text()) )
            else:
                if 'NONE' not in links:
                    links['NONE'] = []
                links['NONE'].append( (a.attr('href'), a.text()) )
        logger.debug('Retrieved data %s %s' % (a.text(), a.attr('href')))

    data = []
    for year in links.keys():
        for link_id, link_name in links[year]:
            logger.info('Getting text at %s' % link_id)
            title = date = text = ''
            for x in d(link_id).parents('p').nextAll().items():
                logger.debug('X: %s' % x.outerHtml())
                if '<a' in x.outerHtml():
                    break
                elif 'End of the Project Gutenberg' in x.text():
                    break
                elif '<h2' in x.outerHtml():
                    title = x.text()
                elif '<h3' in x.outerHtml():
                    date = x.text()
                elif '<p' in x.outerHtml():
                    text += RE_CLEAN_TEXT.sub(x.text().replace('\n',' ').replace('&#13;','').replace('\r',' '), ' ')
                else:
                    logger.error('Unrecognized tag: %s' % x.outerHtml())

            if 'Gutenberg' in text:
                logger.error('%s\n%s' % (title,text))
            logger.debug('\nTitle: %s\nDate: %s\nText: %s' % (title, date, text))
            data.append((year, date, title, text))
    logger.info('Retrieved %d pieces' % len(data))
    return data

def analyze(data, out_dir):
    summary = {}
    freq = FreqDist()
    sentence_length = defaultdict(list)
    year_freq_dist = defaultdict(FreqDist)
    year_dist = defaultdict(int)
    year_month_dist = defaultdict(int)
    year_quarter_dist = defaultdict(int)

    has_date = no_date = sentences = words = 0

    for year, date_str, title, text in data:
        date = parsedate(date_str)
        logger.debug('%s -> %s' % (date_str, str(date)))
        freq.update(ngram_phrases(text,3))
        if date:
            # Since can't use strftime for years before 1900, we need to use isoformat
            year_str = date.isoformat()[:4]
            year_mo_str = date.isoformat()[:7]
            has_date += 1
        else:
            no_date += 1
            year_mo_str = ''

        if year_str:
            year_range = get_year_range(year_str)
            sentence_length[ year_range ].extend( sentence_lengths(text) )
            year_freq_dist[ year_range ].update( ngram_phrases(text,3) )
            year_dist[year] += 1

        if year_mo_str:
            year_month_dist[year_mo_str] += 1
            year_quarter_dist[ year_quarter(year_mo_str) ] += 1

        sentences += count_sentences(text)
        words += count_words(text)

    logger.debug('Documents with a valid date: %d Documents without a valid date: %d' % (has_date, no_date))
    logger.debug('Total # Sentences: %d' % sentences)
    logger.debug('Total $ Words: %d' % words)

    generate_dict_csv(['year', 'cnt'], year_dist, os.path.join(out_dir, 'year-data.csv'))
    generate_dict_csv(['yearmo', 'cnt'], year_month_dist, os.path.join(out_dir, 'year-mo-data.csv'))
    generate_dict_csv(['yearq', 'cnt'], year_quarter_dist, os.path.join(out_dir, 'year-quarter-data.csv'))
    generate_stream_js(year_freq_dist, os.path.join(out_dir, 'stream-data.json'))
    generate_cloud_csv(year_freq_dist, os.path.join(out_dir, 'year-phrase-data.csv'))
    generate_sentence_length_csv(sentence_length, os.path.join(out_dir, 'data-sentence-lengths.csv'))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="data_dir", default=None, help="Data directory", metavar="FILE")
    parser.add_option("-p", "--pickle", dest="pickle_file", default=None, help="Pickle file", metavar="FILE")
    parser.add_option("-o", "--out", dest="out_dir", default=None, help="Out directory", metavar="FILE")

    parser.add_option("-l", "--load", action="store_true", dest="load", default=None, help="Load the data")
    parser.add_option("-a", "--analyze", action="store_true", dest="analyze", default=None, help="Analyze the data")
    (options, args) = parser.parse_args()

    logger.setLevel(logging.DEBUG)

    if options.load:
        files = [ os.path.join(options.data_dir,f) for f in os.listdir(options.data_dir) if f.endswith('.htm') or f.endswith('.html') ]

        logger.info('Retrieved files: %s' % str(files))

        all_data = []
        for f in files:
            logger.info('Processing %s' % f)
            data = process_html_file(f)
            all_data.extend(data)

        logger.info('Pickling')
        output = open(options.pickle_file, 'wb')
        pickle.dump(all_data, output)
        output.close()

    if options.analyze:
        logger.info('Unpickling')
        pkl_file = open(options.pickle_file, 'rb')
        all_data = pickle.load(pkl_file)
        pkl_file.close()

        summary = analyze(all_data, options.out_dir)