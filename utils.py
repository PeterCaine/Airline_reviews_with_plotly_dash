from collections import Counter
import pandas as pd
import pickle
import re
import plotly.graph_objects as go
import plotly.express as px
import gensim
import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.corpus import stopwords

conll_df_dict = pickle.load(open('./data/all_conll_dicts.pkl', 'rb'))
bucket_list_dict = pickle.load(open('./data/bucket_list_dict.pkl', 'rb'))


def load_dfs(airline, min_rating=1, max_rating=5):
    """loads data-frame from pre-processed pickle file

    Args:
        airline (string): name of airline
        min_rating (int): minimum review rating
        max_rating (int): maximum review rating

    Returns:
        data-frame: rating in one column and title and text in another
    """
    df = pickle.load(open(f'./data/dataframe/{airline}.pkl', 'rb'))
    df = df[['rating', 'joined_col']]
    abs_tot = len(df)
    df = df[(df.rating.astype(int) >= int(min_rating)) & (df.rating.astype(int) <= int(max_rating))]
    tot = len(df)
    return df, tot, abs_tot


def keyword_suggest(keyword):
    """uses custom trained Word2vec embeddings to suggest similar words
    (min threshold set to 0.7)

    Args:
        keyword (string): word from list of keywords
    Returns:
        tuple: synonyms for the keyword
    """
    model = gensim.models.Word2Vec.load(
        './data/embeddings/airlines_model')
    synonym_list = (model.wv.most_similar(keyword, topn=5))
    synonym_out = {word[0].lower() for word in synonym_list if word[1] > 0.7}
    return synonym_out


def synonym_selector(word_list, use_syn):
    """calls keyword_suggest function (above) to pass in words and a flag (y/n)
    of whether to use synonyms or not

    Args:
        word_list (list): keywords
        use_syn (string): "y" or "n"

    Returns:
        list: list of keywords either as it was given (if use_syn = 'n') or
        expanded to include synoyms as determined by keyword_suggest (above)
    """
    expanded_wordlist = []
    if use_syn == "y":
        for word in word_list:
            expanded_wordlist.append(word)
            try:
                syn_set = keyword_suggest(word)
                if len(syn_set) > 0:
                    for w in syn_set:
                        expanded_wordlist.append(w)
            except Exception as e:
                print("Oops!", e.__class__, "occurred.")
                pass
    else:

        return word_list
    return expanded_wordlist


def reviews_on_single_keyword(keyword, dataframe):
    """

    :param keyword: (string): user input keyword
    :param dataframe: (pd.DataFrame): selected airline.pkl dataframe of reviews
    :return: dataframe with two columns - rating and review (title and text)
    """
    dataframe['lowered'] = dataframe.joined_col.str.lower()
    kwd = keyword.lower()
    df = dataframe.loc[
                    dataframe.lowered.str.contains(' ' + kwd)]
    return df[['rating', 'joined_col']]


def keyword_reviews(keyword_list, dataframe, use_syn='n'):
    """takes a keyword list and dataframe of pre-processed airline reviews
    returns a list of reviews containing only those keywords (if use_syn =='n')
    or reviews using an expanded range with synonyms for the keywords

    Args:
        keyword_list (list): keywords
        dataframe (df): pre-processed airline reviews
        use_syn (str, optional): a flag to determine whether keyword list is
        expanded or not using synonyms. Defaults to 'n'.

    Returns:
        set: review texts using only the keywords (or expanded with
        synonyms),
        dict: keywords and number of reviews associated to build bar chart
        list: list of keywords with synonyms added (if required)
    """
    dataframe['lowered'] = dataframe.joined_col.str.lower()
    out_dict = {}
    updated_keywords = synonym_selector(keyword_list, use_syn)
    for keyword in updated_keywords:
        keyword = keyword.lower()
        if len(keyword.split(' ')) == 1:
            try:
                out_dict[keyword] = dataframe.loc[
                    dataframe.lowered.str.contains(' ' + keyword + '\W')
                ]
            except Exception as e:
                print("Oops!", e.__class__, "occurred.")
                print(keyword)
        else:
            try:
                out_dict[keyword] = dataframe.loc[
                    dataframe.lowered.str.contains(keyword)
                ]
            except Exception as e:
                print("Oops!", e.__class__, "occurred.")
                print(keyword)
    plotly_dict = {k: len(v) for k, v in out_dict.items()}
    slimmed_dict = {k: v for k, v in out_dict.items() if len(v) > 0}

    tuple_out = [tuple(v[1].itertuples(index=True)) for v in slimmed_dict.items()]
    text_set = set()
    for item in tuple_out:
        for it in item:
            review_text = it[2]
            text_set.add(review_text)
    ct = Counter()
    for values in out_dict.values():
        for texts in values.joined_col:
            ct[texts] += 1
    out_list = ct.most_common()
    overlaps = []
    for each_item in out_list:
        if out_list[0][1] > 1:
            index = 'review #' + str(list(dataframe.index[dataframe.joined_col == each_item[0]])[0])
            item = each_item[0]
            count = each_item[1]
            overlaps.append((index, item, count))
        else:
            overlaps = None
    return list(text_set), plotly_dict, updated_keywords, overlaps


def write_out(airline, texts):
    """

    :param airline: (string) name of airline
    :param texts: (pd.Series) series of selected review texts
    :return: None
    """
    with open(f'./saved_reviews/{airline}_texts.txt', 'a', encoding='utf-8') as outfile:
        for line in texts:
            new_line = line + '\n\n'
            outfile.write(new_line)


def series_to_token_list(series):
    '''
    takes a series of review texts from a dataframe
    returns a concatenated_lowered version with digits replaced by 'dd'
    '''
    # concatenate reviews into single lowered string
    words = [word.lower() for review in series for word in review.split()]
    all_text_as_string = ' '.join(words)
    # replace digits of any length with dd
    digits_stripped = re.sub ('\d+', 'dd', all_text_as_string)
    no_punct = re.sub(r'[^A-Za-z0-9 ] + ', '', digits_stripped)
    # split into list
    out_list = no_punct.split()
    return out_list


def freq_bigram_finder(df, stopwords=stopwords.words('english'), min_freq=4, num_return=50,
                       measure=BigramAssocMeasures.pmi):
    """takes a dataframe and returns most frequent bigrams as measured using a particular metric

    :param df: (pd. DataFrame) dataframe of review texts
    :param stopwords: (set) nltk.corpus.stopwords
    :param min_freq: (int) bigrams are distinct, but too distinct will likely return spelling errors and unwanted noise
    increase frequency to remove noise
    :param num_return: (int) length of list of bigrams out
    :param measure: (bigram metric) PMI
    :return: (string) joined bigrams
    """
    series_as_token_list = series_to_token_list(df['joined_col'])
    bcf = BigramCollocationFinder.from_words(series_as_token_list)
    stopset = set(stopwords)
    filter_stops = lambda w: len(w) < 3 or w in stopset
    bcf.apply_word_filter(filter_stops)
    bcf.apply_freq_filter(min_freq)
    tups_out = bcf.nbest(measure, num_return)
    joined_tups = [' '.join(words) for words in tups_out]
    return joined_tups


def freq_trigram_finder(df, stopwords=stopwords.words('english'), min_freq=4, num_return=50,
                         measure=TrigramAssocMeasures.pmi):
    """takes a dataframe and returns most frequent trigrams as measured using a particular metric

    :param df: (pd. DataFrame) dataframe of review texts
    :param stopwords: (set) nltk.corpus.stopwords
    :param min_freq: (int) trigrams are distinct, but too distinct will likely return spelling errors and unwanted noise
    increase frequency to remove noise
    :param num_return: (int) length of list of trigrams out
    :param measure: (trigram metric) PMI
    :return: (string) joined trigrams
    """
    series_as_token_list = series_to_token_list(df['joined_col'])
    tcf = TrigramCollocationFinder.from_words(series_as_token_list)
    stopset = set(stopwords)
    filter_stops = lambda w: len(w) < 3 or w in stopset
    tcf.apply_word_filter(filter_stops)
    tcf.apply_freq_filter(min_freq)
    tups_out = tcf.nbest(measure, num_return)
    joined_tups = [' '.join(words) for words in tups_out]
    return joined_tups


def count_patterns(airline, num_return=50):
    """produces a list of most_common bigrams based on sheer frequency and their frequency as a string

    :param airline: (string) name of airline
    :param num_return: (int) number of bigrams to return
    :return: string of bigram and count for top (num_return) bigrams
    """
    counter = Counter()
    whole_dict = pickle.load(open('./data/adj_n/all_airlines_adj_n.pkl', 'rb'))
    extracted_pattern_list = whole_dict[airline]
    counter.update(extracted_pattern_list)
    most_common = counter.most_common()
    text_out = ''
    for word, count in most_common[:num_return]:
        text_out += word + ':' + str(count) + ', '
    return text_out


def open_unique_adjn(airline, num_return=50):
    """ given the name of an airline, returns a list of unique noun phrases specific to that airline

    :param airline: (String) name of airline
    :param num_return: (int) number of phrases
    :return: a text a single string of connected unique phrases of the type adj+n
    """
    whole_dict = pickle.load(open('./data/adj_n/unique_five.pkl', 'rb'))
    text_list = list(whole_dict[airline])[:num_return]
    text_out = ' - '.join(text_list)
    return text_out


def load_adj_count(airline, val):
    """ given the name of an airline and a binary value creates:
     - a list of most common adjectives if val = 0
     - a list of most common adjectives unique to that airline if val = 1
     represented in graphical form

    :param airline: (string) name of airline
    :param val: (int) represents boolean 1/0
    :return: a plotly express bar chart representing frequency of adjectives
    """
    count = pickle.load(open(f'./data/adjective_count/adjectives_only_{airline}_counter.pkl', 'rb'))
    x = [tup[0] for tup in count.most_common(val)]
    y = [tup[1] for tup in count.most_common(val)]
    fig = px.bar(x=x, y=y, color=y, title=f"Most Common Adjectives: {airline}", template='plotly_white')
    return fig


def df_lookup_iloc(airline, review_idx):
    """ given the name of an airline and the index of a review, returns the requested review

    :param airline: (string) name of airline
    :param review_idx:
    :return: df.value - review text
    """
    df = pickle.load(open(f'./data/dataframe/{airline}.pkl', 'rb'))
    text_out = df.joined_col.iloc[review_idx]
    return text_out


def load_unique(val):
    """

    :param val: (int) represents boolean 1/0
    :return: dictionary of airline: unique frequent adjectives
    """
    count_klm = pickle.load(open('./data/adjective_count/adjectives_only_KLM_counter.pkl', 'rb'))
    count_aritish_airways = pickle.load(
        open('./data/adjective_count/adjectives_only_British_Airways_counter.pkl', 'rb'))
    count_easyjet = pickle.load(open('./data/adjective_count/adjectives_only_EasyJet_counter.pkl', 'rb'))
    count_ryanair = pickle.load(open('./data/adjective_count/adjectives_only_Ryanair_counter.pkl', 'rb'))
    count_virgin = pickle.load(open('./data/adjective_count/adjectives_only_Virgin_counter.pkl', 'rb'))
    count_dict = {'KLM': count_klm.most_common(),
                  'British_Airways': count_aritish_airways.most_common(),
                  'EasyJet': count_easyjet.most_common(),
                  'Ryanair': count_ryanair.most_common(),
                  'Virgin': count_virgin.most_common()}
    airline_list = ['British_Airways', 'EasyJet', 'KLM', 'Ryanair', 'Virgin']
    final_dict = {}
    for airline in airline_list:
        temp_set = set()
        for k, v in count_dict.items():
            if k != airline:
                for tups in v:
                    temp_set.add(tups[0])
        airline_set = {tup[0] for tup in count_dict[airline]}
        airline_out = airline_set - temp_set
        for_assignment = [v for v in count_dict[airline] if v[0] in airline_out]
        final_dict[airline] = for_assignment[:val]
    return final_dict


def load_tsv(airline_name):
    """loads a tsv file containing CoreNLP parsed reviews in tsv format as pd.DataFrame

    Args:
        airline_name (string): name of airline

    Returns:
        pd.Dataframe: conll formatted reviews as dataframe
    """
    names = ['sent_id', 'word_id', 'word', 'lemma', 'upos', 'xpos', 'head', 'deprel']
    path = './data/conlls/'
    if airline_name == 'Virgin':
        df = pd.read_csv(f'{path}Virgin.tsv', sep='\t', names=names, skiprows=[0])
        return df
    elif airline_name == 'British_Airways':
        df = pd.read_csv(f'{path}British_Airways.tsv', sep='\t', names=names, skiprows=[0])
        return df
    elif airline_name == 'EasyJet':
        df = pd.read_csv(f'{path}EasyJet.tsv', sep='\t', names=names, skiprows=[0])
        return df
    elif airline_name == 'Ryanair':
        df = pd.read_csv(f'{path}Ryanair.tsv', sep='\t', names=names, skiprows=[0])
        return df
    elif airline_name == 'KLM':
        df = pd.read_csv(f'{path}KLM.tsv', sep='\t', names=names, skiprows=[0])
        return df
    else:
        print("no such airline")


def review_ids_for_keyword(airline, keyword):
    """ takes the name of an airline and a keyword and returns a set of review id's

    :param airline: (string) name of airline
    :param keyword: (string) single keyword
    :return: set of review index id's relating to given keyword
    """
    filtered_df, *_ = load_dfs(airline)
    filtered_dff = filtered_df.loc[filtered_df['joined_col'].str.contains(keyword)]
    filtered_dff = filtered_dff.drop_duplicates()
    pure_out = len(filtered_dff)
    return pure_out


def airline_bucket_reviews_count(airline_in):
    """ takes the name of an airline and returns a dict of the airline with counts of reviews containing words
    relating to concept 'buckets'

    :param airline_in: (string) name of airline
    :return: dict of airline: tuples of reviews and length of reviews (number of reviews) for the airline AND key: value
    pais as text for Markdown formatting
    """
    airlines = ['British_Airways', 'EasyJet', 'KLM', 'Ryanair', 'Virgin']
    out_dict = {}
    for airline in airlines:
        interim_dict = {}
        for bucket, bucket_set in bucket_list_dict[airline_in].items():
            df = conll_df_dict[airline]
            dff = df.loc[df['word'].isin(bucket_set)]
            reviews = set(dff.review_id)
            interim_dict[bucket] = (reviews, len(reviews))
        out_dict[airline] = interim_dict
    text_tups_out = []
    for keys, vals in bucket_list_dict[airline_in].items():
        str_vals = [val for val in vals]
        joined_vals = ', '.join(str_vals)
        text_tups_out.append((keys, joined_vals))
    return out_dict, text_tups_out


def graph_display(out_dict, airline):
    """ given a dictionary of airline and reviews associated with a concept, produces data for a bar chart
    comparing number of reviews for 5 airlines

    :param out_dict: (dict) dictionary of airlines and reviews associated with a particular context
    :param airline: (string) name of airline
    :return: plotly go bar chart comparing all 5 airlines
    """
    x = list(bucket_list_dict[airline].keys())
    # since the number of reviews per airline varies, the numerators vary to return a percentage
    ba_y = [tup[1]/149 for tup in out_dict['British_Airways'].values()]
    ez_y = [tup[1]/148 for tup in out_dict['EasyJet'].values()]
    kl_y = [tup[1]/105 for tup in out_dict['KLM'].values()]
    ry_y = [tup[1]/148 for tup in out_dict['Ryanair'].values()]
    vi_y = [tup[1]/145 for tup in out_dict['Virgin'].values()]
    fig = go.Figure(data=[
        go.Bar(name='BA', x=x, y=ba_y),
        go.Bar(name='EZ', x=x, y=ez_y),
        go.Bar(name='KL', x=x, y=kl_y),
        go.Bar(name='RY', x=x, y=ry_y),
        go.Bar(name='VI', x=x, y=vi_y)],
        layout=go.Layout(
        title=go.layout.Title(text=f"{airline} Brand Characteristics Compared")
               )
    )
    fig.update_layout(colorway=['blue', 'orange', 'lightblue', 'darkblue', 'red'],
                      template='plotly_white', yaxis_title="% of Reviews")
    return fig


def context_filter(df_out, keyword, keyword_list=''):
    """ looks for keywords in reviews or if provided a keylist, will look for bigrams involving both the keyword and
    any of the words in the list

    :param df_out: (pd.DataFrame) dataframe of review texts pre-filtered for keyword
    :param keyword: (string) single 'central' keyword as string
    :param keyword_list: (string) one or multiple keywords separated by a comma
    :return: data dict for datatable and columns for names of columns in table & out_list list of concordance lines
    """
    word_list = keyword_list.split(',')
    wd_list = [word.strip() for word in word_list]
    list_out = []
    # this can be easily performed with nltk.bigrams but I wanted to try my hand
    for word in wd_list:
        dff = df_out[df_out.joined_col.str.contains(f'{word} {keyword}')]
        list_out.append(dff)
        dff = df_out[df_out.joined_col.str.contains(f'{keyword} {word}')]
        list_out.append(dff)

    big_df = pd.concat(list_out)
    big_df.drop_duplicates(inplace=True)
    # prep the dcc. (or rather dt) DataTable content from dataframe
    data = big_df.to_dict('records')
    columns = [{"name": i, "id": i, } for i in big_df.columns]
    # produce list of concordance lines from filtered df
    text = big_df.joined_col
    text_block = ' '.join(text)
    text_ls = text_block.split()
    nl_text = nltk.Text(text_ls)
    out_list = nl_text.concordance_list(keyword, width=50, lines=1000)
    return data, columns, out_list
