import argparse
import datetime
from collections import defaultdict, Counter

import dgl
import pandas as pd
import numpy as np
import json
import re
from pprint import pprint


# useful attr: _id, abstract, authors, fos, keywords, n_citations, references, title, venue, year
import torch


def get_lines_data(data_path):
    with open(data_path + 'dblpv13.json') as fr:
        data = fr.read()
        data = re.sub(r"NumberInt\((\d+)\)", r"\1", data)
        print('replace successfully!')
        data = json.loads(data)

    info_fw = open(data_path + 'temp_info.lines', 'w+')
    abstract_fw = open(data_path + 'temp_abstract.lines', 'w+')
    ref_fw = open(data_path + 'temp_ref.lines', 'w+')
    for paper in data:
        info_fw.write(str([paper['_id'],
                           {
                               'authors': paper['authors'] if 'authors' in paper else [],
                               'fos': paper['fos'] if 'fos' in paper else [],
                               'keywords': paper['keywords'] if 'keywords' in paper else [],
                               'n_citations': paper['n_citations'] if 'n_citations' in paper else 0,
                               'title': paper['title'] if 'title' in paper else None,
                               'venue': paper['venue'] if 'venue' in paper else None,
                               'year': paper['year'] if 'year' in paper else None,
                           }]) + '\n')
        temp_ab = paper['abstract'] if 'abstract' in paper else None
        temp_ref = paper['references'] if 'references' in paper else []
        abstract_fw.write(str([paper['_id'], temp_ab]) + '\n')
        ref_fw.write(str([paper['_id'], temp_ref]) + '\n')


def get_all_dict(data_path, part='info'):
    result_dict = {}
    with open(data_path + 'temp_{}.lines'.format(part)) as fr:
        for line in fr:
            data = eval(line)
            result_dict[data[0]] = data[1]
    json.dump(result_dict, open(data_path + 'all_{}_dict.json'.format(part), 'w+'))


# def get_cite_data(data_path):
#     ref_dict = json.load(open(data_path + 'all_ref_dict.json'))
#     cite_dict = defaultdict(list)
#     for paper in ref_dict:
#         # ref_list = ref_dict[paper]
#         for ref in ref_dict[paper]:
#             cite_dict[ref].append(paper)
#     json.dump(cite_dict, open(data_path + 'all_cite_dict.json', 'w+'))
#     del ref_dict
#
#     year_dict = {}
#     info_dict = json.load(open(data_path + 'all_info_dict.json', 'r'))
#     for paper in cite_dict:
#         year_dict[paper] = list(filter(lambda x: x, map(lambda x: info_dict[x]['year'], cite_dict[paper])))
#     json.dump(year_dict, open(data_path + 'cite_year_dict.json', 'w+'))
def get_valid_papers(data_path, time_point=None):
    data = json.load(open(data_path + 'all_info_dict.json', 'r'))
    print(len(data))
    # journal
    data = dict(filter(lambda x: x[1]['venue'].get('_id', False) if x[1]['venue'] else False, data.items()))
    print('has journal:', len(data))

    # authors
    data = dict(filter(lambda x: len([author['_id'] for author in x[1]['authors'] if '_id' in author]) > 0,
                       data.items()))
    print('has author:', len(data))
    # title
    data = dict(filter(lambda x: x[1]['title'], data.items()))
    print('has title:', len(data))
    # json.dump(data, open(data_path + 'sample_info_dict.json', 'w+'))
    selected_list = set(data.keys())
    # time
    if time_point:
        data = dict(filter(lambda x: (int(x[1]['year']) <= time_point), data.items()))
        print('<=year:',len(data))
        # json.dump(data, open(data_path + 'sample_info_dict.json', 'w+'))
        selected_list = set(data.keys())

    del data
    # words
    abs_dict = json.load(open(data_path + 'all_abstract_dict.json', 'r'))
    abs_dict = dict(filter(lambda x: (x[0] in selected_list) & (len(str(x[1]).strip().split(' ')) >= 20),
                           abs_dict.items()))
    selected_list = set(abs_dict.keys())
    print('has abs:', len(selected_list))
    # json.dump(abs_dict, open(data_path + 'sample_abstract_dict.json', 'w+'))
    return set(selected_list)


def get_cite_data(data_path):
    # here already filter the data
    selected_set = get_valid_papers(data_path)
    ref_dict = json.load(open(data_path + 'all_ref_dict.json'))
    cite_dict = defaultdict(list)
    # for paper in ref_dict:
    #     # ref_list = ref_dict[paper]
    #     for ref in ref_dict[paper]:
    #         cite_dict[ref].append(paper)
    for paper in selected_set:
        ref_list = ref_dict.get(paper, [])
        for ref in ref_list:
            if ref in selected_set:
                cite_dict[ref].append(paper)
    json.dump(cite_dict, open(data_path + 'all_cite_dict.json', 'w+'))
    del ref_dict

    year_dict = {}
    info_dict = json.load(open(data_path + 'all_info_dict.json', 'r'))
    for paper in cite_dict:
        year_dict[paper] = list(filter(lambda x: x, map(lambda x: info_dict[x]['year'], cite_dict[paper])))
    json.dump(year_dict, open(data_path + 'cite_year_dict.json', 'w+'))



def show_data(data_path):
    info_dict = json.load(open(data_path + 'all_info_dict.json', 'r'))
    print('total_count', len(info_dict))
    # 'authors', 'fos', 'keywords', 'n_citations', 'title', 'venue', 'year'
    cite = json.load(open(data_path + 'all_cite_dict.json', 'r'))
    dealt_dict = map(lambda x: {
        'id': x[0],
        'authors': len([author['_id'] for author in x[1]['authors'] if '_id' in author]),
        'authors_valid': len([author.get('_id', '') for author in x[1]['authors']
                              if len(author.get('_id', '').strip()) > 0]),
        'authors_g': len([author['gid'] for author in x[1]['authors'] if 'gid' in author]),
        'venue': x[1]['venue'].get('_id', None) if x[1]['venue'] else None,
        'venue_raw': x[1]['venue'].get('raw', None) if x[1]['venue'] else None,
        'fos': len(x[1]['fos']),
        'keywords': len(x[1]['keywords']),
        'year': x[1]['year'],
        'citations': len(cite.get(x[0], []))
    }, info_dict.items())
    no_cite_paper = [paper for paper in info_dict if paper not in cite]
    print(len(no_cite_paper))
    print(len(info_dict))

    df = pd.DataFrame(dealt_dict)
    print(df.describe())
    df_venue = df.groupby('venue').count().sort_values(by='id', ascending=False)['id']
    df_venue.to_csv(data_path + 'stats_venue_count.csv')
    print(df_venue.head(20))
    df_venue_raw = df.groupby(['venue', 'venue_raw']).count().sort_values(by='id', ascending=False)['id']
    df_venue_raw.to_csv(data_path + 'stats_venue_raw_count.csv')
    print(df_venue_raw.head(20))

    df_year = df.groupby('year').count()
    df_year.to_csv(data_path + 'stats_year_count.csv')
    print(df_year)
    del dealt_dict

    abstract_dict = json.load(open(data_path + 'all_abstract_dict.json'))
    # valid_paper = [len(abstract.strip().split(' ')) >= 20 for abstract in abstract_dict.values() if abstract]
    valid_paper = set(map(lambda x: x[0],
                      filter(lambda x: len(str(x[1]).strip().split(' ')) >= 20, abstract_dict.items())))
    print(len(valid_paper))

    df_journal_counts = df.groupby('venue').count().sort_values(by='id', ascending=False)['id']
    df_journal_citations = df.groupby('venue').sum().sort_values(by='citations', ascending=False)['citations']
    print(df_journal_citations.head(20))
    df_journal_avg_citations = (df_journal_citations / df_journal_counts).sort_values(ascending=False)
    print(df_journal_avg_citations.head(20))
    df_journal_counts.to_csv(data_path + 'journal_count.csv')
    df_journal_citations.to_csv(data_path + 'journal_citation.csv')
    df_journal_avg_citations.to_csv(data_path + 'journal_avg_citation.csv')

    df_selected = df[df['id'].isin(valid_paper)]
    df_selected.to_csv(data_path + 'selected_data.csv')
    df_selected_year = df_selected.groupby('year').count()
    df_selected_year.to_csv(data_path + 'stats_selected_year_count.csv')
    print(df_selected_year)


def get_subset(data_path, time_point=None):
    print(time_point)

    selected_set = get_valid_papers(data_path, time_point)
    abs_dict = json.load(open(data_path + 'all_abstract_dict.json', 'r'))
    abs_dict = dict(
        filter(lambda x: (x[0] in selected_set), abs_dict.items()))

    # save subset
    json.dump(abs_dict, open(data_path + 'sample_abstract_dict.json', 'w+'))
    del abs_dict
    data = json.load(open(data_path + 'all_info_dict.json', 'r'))
    data = dict(filter(lambda x: x[0] in selected_set, data.items()))
    json.dump(data, open(data_path + 'sample_info_dict.json', 'w+'))
    del data

    ref = json.load(open(data_path + 'all_ref_dict.json', 'r'))
    # cite = json.load(open(data_path + 'all_cite_data.json', 'r'))
    cite_year = json.load(open(data_path + 'cite_year_dict.json', 'r'))

    ref = dict(filter(lambda x: x[0] in selected_set, ref.items()))
    ref = dict(map(lambda x: (x[0], [paper for paper in x[1] if paper in selected_set]), ref.items()))
    json.dump(ref, open(data_path + 'sample_ref_dict.json', 'w+'))

    cite_year = dict(
        map(lambda x: (x[0], Counter(x[1])), filter(lambda x: x[0] in selected_set, cite_year.items())))
    json.dump(cite_year, open(data_path + 'sample_cite_year_dict.json', 'w+'))

def get_citation_accum(data_path, time_point, all_times=(2011, 2013, 2015)):
    cite_year_path = data_path + 'sample_cite_year_dict.json'
    ref_path = data_path + 'sample_ref_dict.json'
    info_path = data_path + 'sample_info_dict.json'
    cite_year_dict = json.load(open(cite_year_path, 'r'))
    ref_dict = json.load(open(ref_path, 'r'))
    info_dict = json.load(open(info_path, 'r'))

    predicted_list = list(cite_year_dict.keys())
    accum_num_dict = {}

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
    print('writing citations accum')
    pub_dict = {}
    for paper in predicted_list:
        count_dict = dict(map(lambda x: (int(x[0]), x[1]), cite_year_dict[paper].items()))
        if (sum(count_dict.values()) >= 10) and (len(ref_dict[paper]) >= 5):
        # if len(ref_dict[paper]) >= 5:
            # pub_year = int(info_dict[paper]['pub_date']['year'])
            pub_year = int(info_dict[paper]['year'])
            pub_dict[paper] = pub_year
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
    print(len(accum_num_dict))
    # accum_num_dict = dict(filter(lambda x: x[1][-1] >= 10, accum_num_dict.items()))
    print('writing file')
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
    # abs_path = 'all_abstract_dict.json'
    if subset:
        info_path = data_path + 'sample_info_dict.json'
        ref_path = data_path + 'sample_ref_dict.json'
        cite_year_path = data_path + 'sample_cite_year_dict.json'
        # abs_path = 'sample_abstract_dict.json'

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

    author_trans = {}
    journal_trans = {}
    author_src, author_dst = [], []
    journal_src, journal_dst = [], []
    author_index = 0
    journal_index = 0
    for paper in paper_trans:
        meta_data = info_dict[paper]
        try:
            # authors = list(map(lambda x: (x['name'].get('given-names', '') + ' ' + x['name'].get('sur', '')).strip().lower(),
            #               meta_data['authors']))
            authors = set([author['_id'] for author in meta_data['authors'] if '_id' in author])
        except Exception as e:
            print(meta_data['authors'])
            authors = None
        # print(authors)
        # print(meta_data['authors'][0]['name'])
        # journal = meta_data['journal']['ids']['nlm-ta'] if 'nlm-ta' in meta_data['journal']['ids'] else None
        journal = meta_data['venue']['_id'] if '_id' in meta_data['venue'] else None

        for author in authors:
            if author in author_trans:
                cur_idx = author_trans[author]
            else:
                cur_idx = author_index
                author_trans[author] = cur_idx
                author_index += 1
            author_src.append(cur_idx)
            author_dst.append(paper_trans[paper])

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
    # paper_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['pub_date']['year']) for paper in dst_list]
    # author_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['pub_date']['year']) for paper in author_dst]
    # journal_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['pub_date']['year']) for paper in journal_dst]
    paper_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in dst_list]
    author_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in author_dst]
    journal_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in journal_dst]
    paper_time = [int(info_dict[paper]['year']) for paper in node_trans['paper'].keys()]

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

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    parser.add_argument('--data_path', default=None, help='the input.')
    parser.add_argument('--name', default=None, help='file name.')
    parser.add_argument('--out_path', default=None, help='the output.')
    # parser.add_argument('--seed', default=123, help='the seed.')
    args = parser.parse_args()
    # TIME_POINT = 2015
    # ALL_TIMES = (2011, 2013, 2015)
    TIME_POINT = 2013
    ALL_TIMES = (2009, 2011, 2013)
    if args.phase == 'test':
        print('This is a test process.')
        # get_lines_data('./data/')
        # get_all_dict('./data/', 'info')
        # get_all_dict('./data/', 'abstract')
        # get_all_dict('./data/', 'ref')
        # get_cite_data('../data/')
    elif args.phase == 'lines_data':
        get_lines_data(args.data_path)
        print('lines data done')
    elif args.phase == 'info_data':
        get_all_dict(args.data_path, 'info')
        print('nfo data done')
    elif args.phase == 'abstract_data':
        get_all_dict(args.data_path, 'abstract')
        print('abstract data done')
    elif args.phase == 'ref_data':
        get_all_dict(args.data_path, 'ref')
        print('ref data done')
    elif args.phase == 'cite_data':
        get_cite_data(args.data_path)
        print('cite data done')
    elif args.phase == 'show_data':
        show_data(args.data_path)
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