import argparse
import datetime
import os
import pandas as pd
from bs4 import BeautifulSoup, Tag
import json
from collections import Counter
import dgl
import torch


def get_valid_ref_list(soup):
    result_dict = {}
    ref_list = soup.find('ref-list')
    if ref_list:
        ref_list = ref_list.find_all('ref')
        for ref in ref_list:
            ref_label = ref.attrs['id']
            ref_ids = ref.find_all('pub-id')
            if len(ref_ids) < 1:
                continue
            #         print(ref_ids)
            result_dict[ref_label] = {ref_id.attrs['pub-id-type']: ref_id.text for ref_id in ref_ids}
    return result_dict


def get_name(contrib):
    if contrib.find('name'):
        return {name.name: name.text for name in contrib.find('name').children if isinstance(name, Tag)}
    else:
        return None


def get_id(contrib):
    return {contrib_id.attrs['contrib-id-type']: contrib_id.text for contrib_id in contrib.find_all('contrib-id')}


def get_article_meta(soup):
    article = soup.front.find('article-meta')
    if article.find('kwd-group'):
        kwds = [kwd.text for kwd in article.find('kwd-group').find_all('kwd')]
    else:
        kwds = []

    if article.find('abstract'):
        abstract = article.find('abstract').text
    else:
        abstract = ''

    pub_date = {'year': article.find('pub-date').find('year').text if article.find('pub-date').find('year') else None,
                'month': article.find('pub-date').find('month').text if article.find('pub-date').find(
                    'month') else None,
                'day': article.find('pub-date').find('day').text if article.find('pub-date').find('day') else None}

    if article.find('contrib-group'):
        authors = article.find('contrib-group').find_all('contrib')
        if authors:
            # authors = [{'type': contrib.attrs, 'name': {name.name: name.text for name in contrib.find('name').children
            #                                             if isinstance(name, Tag)}}
            #            for contrib in authors if contrib.find('name')]
            authors = [{'type': contrib.attrs, 'name': get_name(contrib), 'contrib-id': get_id(contrib)}
                       for contrib in authors]
        else:
            authors = []
        aff = [str(item) for item in article.find('contrib-group').find_all('aff')]  # 没想好怎么处理机构
    else:
        authors = []
        aff = []

    cat = article.find('article-categories').subject.text if article.find('article-categories') else None
    ids = {arc_id.attrs['pub-id-type']: arc_id.text for arc_id in article.find_all('article-id')}
    title = article.find('article-title').text if article.find('article-title') else None

    journal = soup.front.find('journal-meta')
    if journal:
        journal_ids = {jid.attrs['journal-id-type']: jid.text for jid in journal.find_all('journal-id')}
        journal_title = journal.find('journal-title').text if journal.find('journal-title') else None
    else:
        journal_ids = {}
        journal_title = None

    if soup.body:
        full = len(soup.body.find_all('xref', attrs={'ref-type': 'bibr'})) > 0
    else:
        full = False

    return {
        'pmc': ids['pmc'],
        'ids': ids,
        'title': title,
        'cat': cat,
        'aff': aff,
        'authors': authors,
        'pub_date': pub_date,
        'abstract': abstract,
        'kwds': kwds,
        'journal': {'ids': journal_ids, 'title': journal_title},
        'full': full
    }


def get_simple_data(data_path, out_path):
    ref_dict = {}
    pub_dict = {}
    folder_list = os.listdir(data_path)
    for folder in folder_list:
        folder_path = data_path + folder
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                file_path = folder_path + '/' + file
                #         print(file_path)
                try:
                    with open(file_path) as fr:
                        sample = fr.readlines()
                    soup = BeautifulSoup(''.join(sample), ["lxml-xml"])

                    temp_ref = get_valid_ref_list(soup)
                    temp_pub = get_article_meta(soup)
                    cur_pmc = temp_pub['pmc']
                    ref_dict[cur_pmc] = temp_ref
                    pub_dict[cur_pmc] = temp_pub
                except Exception as e:
                    print(e)
                    print(file_path)
                finally:
                    continue
    json.dump(ref_dict, open('{}.ref'.format(out_path), 'w+', encoding='utf-8'))
    json.dump(pub_dict, open('{}.pub'.format(out_path), 'w+', encoding='utf-8'))


def get_lines_data(data_path, out_path):
    folder_list = os.listdir(data_path)
    ref_fr = open('{}_lines.ref'.format(out_path), 'w+', encoding='utf-8')
    pub_fr = open('{}_lines.pub'.format(out_path), 'w+', encoding='utf-8')
    count = 0
    for folder in folder_list:
        folder_path = data_path + folder
        files = os.listdir(folder_path)
        for file in files:
            file_path = folder_path + '/' + file
            #         print(file_path)
            count += 1
            try:
                with open(file_path) as fr:
                    sample = fr.readlines()
                soup = BeautifulSoup(''.join(sample), ["lxml-xml"])
                temp_ref = get_valid_ref_list(soup)
                temp_pub = get_article_meta(soup)
                temp_pub['file_path'] = file_path
                cur_pmc = temp_pub['pmc']
                ref_fr.write(str({'pmc': cur_pmc, 'ref': temp_ref}) + '\n')
                pub_fr.write(str({'pmc': cur_pmc, 'pub': temp_pub}) + '\n')
            except Exception as e:
                print(e)
                print(file_path)
            finally:
                continue


def get_pub_data(data_path):
    # words more than 20
    pub_dict = {}
    files_list = os.listdir(data_path)
    files_list = [file for file in files_list if file.endswith('lines.pub')]
    del_list = []
    for file in files_list:
        print(file)
        with open(data_path + file, 'r', encoding='utf-8') as fr:
            temp_data = list(map(lambda x: eval(x), fr.readlines()))
        for paper in temp_data:
            if len(paper['pub']['abstract'].strip().split(' ')) >= 20:
                pub_dict[paper['pmc']] = paper['pub']
            else:
                print('del ' + paper['pmc'])
                del_list.append(paper['pmc'])
                continue
    json.dump(pub_dict, open(data_path + 'all_pub_data.json', 'w+'))

    doi_trans = {}
    pmid_trans = {}
    for paper in pub_dict.values():
        if 'pmid' in paper['ids']:
            pmid_trans[paper['ids']['pmid']] = paper['pmc']
        if 'doi' in paper['ids']:
            doi_trans[paper['ids']['doi']] = paper['pmc']
    json.dump(pmid_trans, open(data_path + 'pmid_trans.json', 'w+'))
    json.dump(doi_trans, open(data_path + 'doi_trans.json', 'w+'))
    print('del num:', len(del_list))

    with open(data_path + 'all_ids.list', 'w+') as fw:
        fw.writelines([pmc + '\n' for pmc in pub_dict])


def get_valid_papers(data_path, time_point=None):
    data = json.load(open(data_path + 'all_pub_data.json', 'r'))
    print(len(data))
    # words more than 20
    data = dict(filter(lambda x: len(x[1]['abstract'].strip().split(' ')) >= 20, data.items()))
    data = dict(filter(lambda x: x[1]['title'], data.items()))
    print('has abs+title:', len(data))
    # data = dict(filter(lambda x: (int(x[1]['pub_date']['year']) <= time_point) |
    #                              ((len(cite[x[1]['pmc']]) > 0) & (int(x[1]['pub_date']['year']) < time_range[1]))
    #                    # & (x[1]['journal']['title'] in selected_journal_list)
    #                    , data.items()))
    # data = dict(filter(lambda x: x[1]['journal'].get('title', '').lower() in selected_journal, data.items()))
    # data = dict(filter(lambda x: x[1]['journal']['ids'].get('nlm-ta', '') in selected_journal, data.items()))
    data = dict(filter(lambda x: x[1]['journal']['ids'].get('nlm-ta', False), data.items()))
    print('has journal:', len(data))
    if time_point:
        data = dict(filter(lambda x: (int(x[1]['pub_date']['year']) <= time_point), data.items()))
        print('<={}:'.format(time_point), len(data))
    return set(data.keys())


def get_ref_data(data_path):
    ref_dict = {}
    files_list = os.listdir(data_path)
    files_list = [file for file in files_list if file.endswith('lines.ref')]
    doi_trans = json.load(open(data_path + 'doi_trans.json'))
    pmid_trans = json.load(open(data_path + 'pmid_trans.json'))

    print('loading data...')
    # with open(data_path + 'all_ids.list', 'r+') as fr:
    #     all_paper_pmc = set(list(map(lambda x: x.strip(), fr.readlines())))
    all_paper_pmc = get_valid_papers(data_path)
    for file in files_list:
        print(file)
        with open(data_path + file, 'r', encoding='utf-8') as fr:
            temp_data = list(map(lambda x: eval(x), fr.readlines()))
        for paper in temp_data:
            if paper['pmc'] in all_paper_pmc:
                ref_dict[paper['pmc']] = paper['ref']
            else:
                continue

    dealt_ref = {}
    print('dealting data...')
    for paper in ref_dict:
        ref_list = []
        temp_ref = ref_dict[paper].values()
        for ref_paper in temp_ref:
            cur_id = None
            if 'pmcid' in ref_paper:
                if ref_paper['pmcid'] in all_paper_pmc:
                    # ref_list.append(ref_paper['pmcid'])
                    cur_id = ref_paper['pmcid']
            elif 'pmid' in ref_paper:
                if ref_paper['pmid'] in pmid_trans:
                    # ref_list.append(pmid_trans[ref_paper['pmid']])
                    cur_id = pmid_trans[ref_paper['pmid']]
            elif 'doi' in ref_paper:
                if ref_paper['doi'] in doi_trans:
                    # ref_list.append(doi_trans[ref_paper['doi']])
                    cur_id = doi_trans[ref_paper['doi']]
            if cur_id in all_paper_pmc:
                ref_list.append(cur_id)
        dealt_ref[paper] = ref_list

    json.dump(dealt_ref, open(data_path + 'all_ref_data.json', 'w+'))

    print('dealting citing data...')
    cite_dict = {paper: [] for paper in all_paper_pmc}
    for (paper, ref_list) in dealt_ref.items():
        for ref_paper in ref_list:
            cite_dict[ref_paper].append(paper)
    json.dump(cite_dict, open(data_path + 'all_cite_data.json', 'w+'))

    del ref_dict

    year_dict = {}
    pub_dict = json.load(open(data_path + 'all_pub_data.json', 'r'))
    for paper in cite_dict:
        year_dict[paper] = list(map(lambda x: int(pub_dict[x]['pub_date']['year']), cite_dict[paper]))
    json.dump(year_dict, open(data_path + 'cite_year_data.json', 'w+'))


def show_data(data_path, time_range=None):
    data = json.load(open(data_path + 'all_pub_data.json', 'r'))
    ref = json.load(open(data_path + 'all_ref_data.json', 'r'))
    cite = json.load(open(data_path + 'all_cite_data.json', 'r'))
    print(list(data.values())[0])
    print(list(data.values())[0].keys())
    data = map(lambda x: {
        'pmc': x['pmc'],
        'type': x['cat'],
        'aff': len(x['aff']),
        'authors': len(x['authors']),
        'pub_date': int(x['pub_date']['year']),
        'abstract': len(x['abstract'].strip().split(' ')) >= 20,
        'kwds': len(x['kwds']),
        'journal': x['journal']['title'].lower() if 'title' in x['journal'] else None,
        'journal_id': x['journal']['ids']['nlm-ta'] if 'nlm-ta' in x['journal']['ids'] else None,
        'full': x['full'],
        'ref': len(ref[x['pmc']]),
        'citations': len(cite[x['pmc']])
    }, data.values())

    df = pd.DataFrame(data)
    if time_range:
        print(df.shape)
        df = df[(df['pub_date'] >= time_range[0]) & (df['pub_date'] < time_range[1])]
        df.groupby('type').count().sort_values(by='pmc', ascending=False)['pmc'] \
            .to_csv(data_path + 'stats_type_count_{}_{}.csv'.format(time_range[0], time_range[1]))
        df.groupby('pub_date').count()['pmc'] \
            .to_csv(data_path + 'stats_year_count_{}_{}.csv'.format(time_range[0], time_range[1]))
        print('good_paper:',
              df[df.apply(lambda x: (x['ref'] >= 5) & (x['citations'] > 0) & (x['abstract']) & (x['kwds'] > 0),
                          axis=1)].shape[0])
        print('connected_paper:',
              df[df.apply(lambda x: (x['ref'] > 0) & (x['citations'] > 0) & (x['abstract']), axis=1)].shape[0])
        print('connected_paper with kwds:',
              df[df.apply(lambda x: (x['ref'] > 0) & (x['citations'] > 0) & (x['abstract']) & (x['kwds'] > 0),
                          axis=1)].shape[0])
        print('leaf_paper:',
              df[df.apply(lambda x: (x['ref'] > 0) & (x['citations'] == 0) & (x['abstract']), axis=1)].shape[0])
        print('leaf_paper with kwds:',
              df[df.apply(lambda x: (x['ref'] > 0) & (x['citations'] == 0) & (x['abstract']) & (x['kwds'] > 0),
                          axis=1)].shape[0])
    else:
        df.groupby('type').count().sort_values(by='pmc', ascending=False)['pmc'] \
            .to_csv(data_path + 'stats_type_count.csv')
        df.groupby('pub_date').count()['pmc'].to_csv(data_path + 'stats_year_count.csv')
    print(df.shape)
    print(df.columns)
    print(df.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]))
    print(df.groupby('type').count().sort_values(by='pmc', ascending=False)['pmc'].head(20))
    print(df.groupby('pub_date').count().sort_values(by='pmc', ascending=False)['pmc'].head(20))
    print(df.groupby('abstract').count().sort_values(by='pmc', ascending=False)['pmc'].head(20))
    print(df.groupby('journal').count().sort_values(by='pmc', ascending=False)['pmc'].head(20))
    print(df.groupby('full').count().sort_values(by='pmc', ascending=False)['pmc'].head(20))
    # df_journal_counts = df.groupby('journal').count().sort_values(by='pmc', ascending=False)['pmc']
    # df_journal_citations = df.groupby('journal').sum().sort_values(by='citations', ascending=False)['citations']
    # print(df_journal_citations.head(20))
    # df_journal_avg_citations = (df_journal_citations / df_journal_counts).sort_values(ascending=False)
    # print(df_journal_avg_citations.head(20))
    # df_journal_counts.to_csv(data_path + 'journal_count_{}_{}.csv'.format(time_range[0], time_range[1]))
    # df_journal_citations.to_csv(data_path + 'journal_citation_{}_{}.csv'.format(time_range[0], time_range[1]))
    # df_journal_avg_citations.to_csv(data_path + 'journal_avg_citation_{}_{}.csv'.format(time_range[0], time_range[1]))
    df_journal_counts = df.groupby('journal_id').count().sort_values(by='pmc', ascending=False)['pmc']
    df_journal_citations = df.groupby('journal_id').sum().sort_values(by='citations', ascending=False)['citations']
    print(df_journal_citations.head(20))
    df_journal_avg_citations = (df_journal_citations / df_journal_counts).sort_values(ascending=False)
    print(df_journal_avg_citations.head(20))
    df_journal_counts.to_csv(data_path + 'journal_count_{}_{}.csv'.format(time_range[0], time_range[1]))
    df_journal_citations.to_csv(data_path + 'journal_citation_{}_{}.csv'.format(time_range[0], time_range[1]))
    df_journal_avg_citations.to_csv(data_path + 'journal_avg_citation_{}_{}.csv'.format(time_range[0], time_range[1]))
    df.groupby(['journal_id', 'journal']).count().sort_values(by='pmc', ascending=False)['pmc'].to_csv(data_path +
                                                                                                       'journal_id.csv')


def get_subset(data_path, time_point=None):
    print(time_point)
    data = json.load(open(data_path + 'all_pub_data.json', 'r'))
    ref = json.load(open(data_path + 'all_ref_data.json', 'r'))
    # cite = json.load(open(data_path + 'all_cite_data.json', 'r'))
    cite_year = json.load(open(data_path + 'cite_year_data.json', 'r'))
    selected_set = get_valid_papers(data_path, time_point)
    data = dict(filter(lambda x: x[0] in selected_set, data.items()))
    json.dump(data, open(data_path + 'sample_info_dict.json', 'w+'))

    ref = dict(filter(lambda x: x[0] in selected_set, ref.items()))
    ref = dict(map(lambda x: (x[0], [paper for paper in x[1] if paper in selected_set]), ref.items()))
    json.dump(ref, open(data_path + 'sample_ref_dict.json', 'w+'))

    cite_year = dict(
        map(lambda x: (x[0], Counter(x[1])), filter(lambda x: x[0] in selected_set, cite_year.items())))
    json.dump(cite_year, open(data_path + 'sample_cite_year_dict.json', 'w+'))


def get_citation_accum(data_path, time_point=None, all_times=(2011, 2013, 2015)):
    info_path = data_path + 'sample_info_dict.json'
    ref_path = data_path + 'sample_ref_dict.json'
    cite_year_path = data_path + 'sample_cite_year_dict.json'

    info_dict = json.load(open(info_path, 'r'))
    ref_dict = json.load(open(ref_path, 'r'))
    cite_year_dict = json.load(open(cite_year_path, 'r'))
    predicted_list = list(cite_year_dict.keys())
    accum_num_dict = {}
    print('writing citations accum')
    valid_time_dict = {}
    pub_dict = {}
    for paper in predicted_list:
        count_dict = dict(map(lambda x: (int(x[0]), x[1]), cite_year_dict[paper].items()))
        if (sum(count_dict.values()) >= 10) and (len(ref_dict[paper]) >= 5):
            # if (sum(count_dict.values()) >= 10):
            pub_year = int(info_dict[paper]['pub_date']['year'])
            pub_dict[paper] = pub_year
            valid_time_dict[paper] = pub_year
            start_year = max(pub_year, time_point - 4)
            # print(count_dict)
            count = sum(dict(filter(lambda x: x[0] <= start_year, count_dict.items())).values())
            # print(start_count)
            # temp_input_accum_num = [None] * (start_year + 4 - time_point)
            temp_input_accum_num = [-1] * (start_year + 4 - time_point)
            temp_output_accum_num = []
            for year in range(start_year, time_point + 1):
                if year in count_dict:
                    count += int(count_dict[year])
                temp_input_accum_num.append(count)
            for year in range(time_point + 1, time_point + 6):
                if year in count_dict:
                    count += int(count_dict[year])
                temp_output_accum_num.append(count)
            accum_num_dict[paper] = (temp_input_accum_num, temp_output_accum_num)
            # print(temp_input_accum_num)
    print('all_papers:', len(accum_num_dict))
    # df_time = pd.DataFrame(valid_time_dict.items())
    # df_time.columns = ['id', 'year']
    # print(df_time.head())
    # print('2011:', df_time[df_time['year'] <= 2011].shape[0])
    # print('2013:', df_time[df_time['year'] <= 2013].shape[0])
    # print('2015:', df_time[df_time['year'] <= 2015].shape[0])
    json.dump(accum_num_dict, open(data_path + 'sample_citation_accum.json', 'w+'))
    for cur_time in all_times:
        cur_dict = {}
        cur_papers = map(lambda x: x[0], filter(lambda x: x[1] <= cur_time, pub_dict.items()))
        for paper in cur_papers:
            cur_dict[paper] = accum_num_dict[paper][1][cur_time + 4 - time_point] - \
                              accum_num_dict[paper][0][cur_time + 4 - time_point]
        print(cur_time, len(cur_dict))
        df = pd.DataFrame(data=cur_dict.items())
        df.columns = ['paper', 'citations']
        df.to_csv(data_path + 'citation_{}.csv'.format(cur_time))



def get_input_data(data_path, time_point=None, subset=False):
    info_path = data_path + 'all_info_dict.json'
    ref_path = data_path + 'all_ref_dict.json'
    cite_year_path = data_path + 'all_cite_year_dict.json'
    if subset:
        info_path = data_path + 'sample_info_dict.json'
        ref_path = data_path + 'sample_ref_dict.json'
        cite_year_path = data_path + 'sample_cite_year_dict.json'

    info_dict = json.load(open(info_path, 'r'))
    ref_dict = json.load(open(ref_path, 'r'))
    cite_year_dict = json.load(open(cite_year_path, 'r'))

    node_trans = {}
    paper_trans = {}
    # graph created
    src_list = []
    dst_list = []
    index = 0
    for dst in ref_dict:
        if dst in paper_trans:
            dst_idx = paper_trans[dst]
        else:
            dst_idx = index
            paper_trans[dst] = dst_idx
            index += 1
        for src in set(ref_dict[dst]):
            if src in paper_trans:
                src_idx = paper_trans[src]
            else:
                src_idx = index
                paper_trans[src] = src_idx
                index += 1
            src_list.append(src_idx)
            dst_list.append(dst_idx)

    node_trans['paper'] = paper_trans
    # remove author
    # for author related cascade model, use paper id instead

    author_trans = {}
    journal_trans = {}
    author_src, author_dst = [], []
    journal_src, journal_dst = [], []
    # author_index = 0
    journal_index = 0
    for paper in paper_trans:
        meta_data = info_dict[paper]
        # try:
        #     authors = list(
        #         map(lambda x: (x['name'].get('given-names', '') + ' ' + x['name'].get('sur', '')).strip().lower(),
        #             meta_data['authors']))
        # except Exception as e:
        #     print(list(map(lambda x: x['name'], meta_data['authors'])))
        #     authors = None
        # print(authors)
        # print(meta_data['authors'][0]['name'])
        journal = meta_data['journal']['ids']['nlm-ta'] if 'nlm-ta' in meta_data['journal']['ids'] else None

        # for author in authors:
        #     if author in author_trans:
        #         cur_idx = author_trans[author]
        #     else:
        #         cur_idx = author_index
        #         author_trans[author] = cur_idx
        #         author_index += 1
        #     author_src.append(cur_idx)
        #     author_dst.append(paper_trans[paper])

        if journal:
            if journal in journal_trans:
                cur_idx = journal_trans[journal]
            else:
                cur_idx = journal_index
                journal_trans[journal] = cur_idx
                journal_index += 1
            journal_src.append(cur_idx)
            journal_dst.append(paper_trans[paper])

    node_trans['author'] = author_trans
    node_trans['journal'] = journal_trans

    # node_trans_reverse = dict(map(lambda x: (x[0], dict(zip(x[1].values(), x[1].keys()))), node_trans.items()))
    node_trans_reverse = {key: dict(zip(node_trans[key].values(), node_trans[key].keys())) for key in node_trans}
    paper_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['pub_date']['year']) for paper in dst_list]
    author_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['pub_date']['year']) for paper in author_dst]
    journal_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['pub_date']['year']) for paper in
                         journal_dst]
    paper_time = [int(info_dict[paper]['pub_date']['year']) for paper in node_trans['paper'].keys()]

    if subset:
        json.dump(node_trans, open(data_path + 'sample_node_trans.json', 'w+'))
    else:
        json.dump(node_trans, open(data_path + 'all_node_trans.json', 'w+'))

    # graph = dgl.graph((src_list, dst_list), num_nodes=len(paper_trans))
    graph = dgl.heterograph({
        ('paper', 'is cited by', 'paper'): (src_list, dst_list),
        ('paper', 'cites', 'paper'): (dst_list, src_list),
        ('author', 'writes', 'paper'): (author_src, author_dst),
        ('paper', 'is writen by', 'author'): (author_dst, author_src),
        ('journal', 'publishes', 'paper'): (journal_src, journal_dst),
        ('paper', 'is published by', 'journal'): (journal_dst, journal_src),
    }, num_nodes_dict={
        'paper': len(paper_trans),
        'author': len(author_trans),
        'journal': len(journal_trans)
    })
    # graph.ndata['paper_id'] = torch.tensor(list(node_trans.keys())).unsqueeze(dim=0)
    graph.nodes['paper'].data['time'] = torch.tensor(paper_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is cited by'].data['time'] = torch.tensor(paper_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['cites'].data['time'] = torch.tensor(paper_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['writes'].data['time'] = torch.tensor(author_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is writen by'].data['time'] = torch.tensor(author_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['publishes'].data['time'] = torch.tensor(journal_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is published by'].data['time'] = torch.tensor(journal_link_time, dtype=torch.int16).unsqueeze(dim=-1)

    graph = dgl.remove_self_loop(graph, 'is cited by')
    graph = dgl.remove_self_loop(graph, 'cites')
    print(graph)
    print(graph.edges['cites'].data['time'])
    if subset:
        torch.save(graph, data_path + 'graph_sample')
    else:
        torch.save(graph, data_path + 'graph')

    del graph, src_list, dst_list

    # predicted_list = list(cite_year_dict.keys())
    # accum_num_dict = {}

    # for paper in predicted_list:
    #     temp_cum_num = []
    #     pub_year = int(info_dict[paper]['pub_date']['year'])
    #     count_dict = cite_year_dict[paper]
    #     count = 0
    #     for year in range(pub_year, pub_year+time_window):
    #         if str(year) in count_dict:
    #             count += int(count_dict[str(year)])
    #         temp_cum_num.append(count)
    #     accum_num_dict[paper] = temp_cum_num
    #
    # print(len(accum_num_dict))
    # print(len(list(filter(lambda x: x[-1] > 0, accum_num_dict.values()))))
    # print(len(list(filter(lambda x: x[-1] >= 10, accum_num_dict.values()))))
    # print(len(list(filter(lambda x: x[-1] >= 100, accum_num_dict.values()))))
    #
    # if subset:
    #     accum_num_dict = dict(filter(lambda x: x[1][-1] >= 10, accum_num_dict.items()))
    #     json.dump(accum_num_dict, open(data_path + 'sample_citation_accum.json', 'w+'))
    # else:
    #     json.dump(accum_num_dict, open(data_path + 'all_citation_accum.json', 'w+'))
    # print('writing citations accum')
    # valid_time_dict = {}
    # for paper in predicted_list:
    #     count_dict = dict(map(lambda x: (int(x[0]), x[1]), cite_year_dict[paper].items()))
    #
    #     if (sum(count_dict.values()) >= 10) and (len(ref_dict[paper]) >= 5):
    #     # if (sum(count_dict.values()) >= 10):
    #         pub_year = int(info_dict[paper]['pub_date']['year'])
    #         valid_time_dict[paper] = pub_year
    #         start_year = max(pub_year, time_point - 4)
    #         # print(count_dict)
    #         count = sum(dict(filter(lambda x: x[0] <= start_year, count_dict.items())).values())
    #         # print(start_count)
    #         # temp_input_accum_num = [None] * (start_year + 4 - time_point)
    #         temp_input_accum_num = [-1] * (start_year + 4 - time_point)
    #         temp_output_accum_num = []
    #         for year in range(start_year, time_point + 1):
    #             if year in count_dict:
    #                 count += int(count_dict[year])
    #             temp_input_accum_num.append(count)
    #         for year in range(time_point + 1, time_point + 6):
    #             if year in count_dict:
    #                 count += int(count_dict[year])
    #             temp_output_accum_num.append(count)
    #         accum_num_dict[paper] = (temp_input_accum_num, temp_output_accum_num)
    #         # print(temp_input_accum_num)
    # print('all_papers:', len(accum_num_dict))
    # df_time = pd.DataFrame(valid_time_dict.items())
    # df_time.columns = ['id', 'year']
    # print(df_time.head())
    # print('2011:', df_time[df_time['year'] <= 2011].shape[0])
    # print('2013:', df_time[df_time['year'] <= 2013].shape[0])
    # print('2015:', df_time[df_time['year'] <= 2015].shape[0])
    # if subset:
    #     # accum_num_dict = dict(filter(lambda x: x[1][-1] >= 10, accum_num_dict.items()))
    #     print('writing file')
    #     json.dump(accum_num_dict, open(data_path + 'sample_citation_accum.json', 'w+'))
    # else:
    #     json.dump(accum_num_dict, open(data_path + 'all_citation_accum.json', 'w+'))


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    parser.add_argument('--data_path', default=None, help='the input.')
    parser.add_argument('--out_path', default=None, help='out path and file name.')
    # parser.add_argument('--out_path', default=None, help='the output.')
    # parser.add_argument('--seed', default=123, help='the seed.')
    args = parser.parse_args()
    TIME_POINT = 2015
    ALL_TIMES = (2011, 2013, 2015)
    if args.phase == 'test':
        print('This is a test process.')
        # get_simple_data('../data/pubmed/sample/', '../data/pubmed/sample')
        # get_lines_data('../data/pubmed/sample/', '../data/pubmed/sample')
        # get_pub_data('../data/pubmed/')
        get_ref_data('../data/pubmed/')
        # show_data('../data/pubmed/', [1980, 2021])
        # get_subset('../data/pubmed/', TIME_POINT)
        # get_citation_accum('../data/pubmed/', TIME_POINT, ALL_TIMES)
        # get_input_data('../data/pubmed/', time_point=TIME_POINT, subset=True)
    elif args.phase == 'simple_data':
        get_simple_data(args.data_path, args.out_path)
        print('simple data done')
    elif args.phase == 'lines_data':
        get_lines_data(args.data_path, args.out_path)
        print('lines data done')
    elif args.phase == 'pub_data':
        get_pub_data(args.data_path)
        print('pub data done')
    elif args.phase == 'ref_data':
        get_ref_data(args.data_path)
        print('ref data done')
    elif args.phase == 'show_data':
        show_data(args.data_path, [1980, 2011])
        print('show data done')
    elif args.phase == 'subset':
        get_subset(args.data_path, time_point=TIME_POINT)
        print('subset data done')
    elif args.phase == 'subset_input_data':
        get_input_data(args.data_path, subset=True, time_point=TIME_POINT)
        print('subset input data done')
    elif args.phase == 'citation_accum':
        get_citation_accum(args.data_path, time_point=TIME_POINT, all_times=ALL_TIMES)
        print('citation accum done')