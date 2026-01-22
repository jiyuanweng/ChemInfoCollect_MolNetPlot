'''
adopted and modified from following packages:
https://github.com/JamesJeffryes/pyclassyfire
https://pypi.org/project/pybatchclassyfire/ 
'''

import re
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
import json
import numpy as np

#region clean classyfire response json

def _get_name_value(obj):
    result = []
    
    if isinstance(obj, dict):
        if 'name' in obj:
            result.append(obj['name'])
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and 'name' in item:
                result.append(item['name'])

    return result

def _clean_to_table(jsondic):
    
    mdict = {}
    
    if (jsondic != {} and jsondic != None):

        if sum([bool(re.match('inchikey', x)) for x in jsondic.keys()]) > 0 and jsondic['inchikey'] is not None:
            mdict['inchikey'] = jsondic['inchikey'].split("=")[1]
        if sum([bool(re.match('smiles', x)) for x in jsondic.keys()]) > 0 and jsondic['smiles'] is not None:
            mdict['smiles'] = jsondic['smiles']
        if sum([bool(re.match('kingdom', x)) for x in jsondic.keys()]) > 0 and jsondic['kingdom'] is not None:
            mdict['kingdom'] = jsondic['kingdom']['name']
        if sum([bool(re.match('superclass', x)) for x in jsondic.keys()]) > 0 and jsondic['superclass'] is not None :
            mdict['superclass'] = jsondic['superclass']['name']
        if sum([bool(re.match('class', x)) for x in jsondic.keys()]) > 0 and jsondic['class'] is not None :
            mdict['class'] = jsondic['class']['name']
        if sum([bool(re.match('subclass', x)) for x in jsondic.keys()]) > 0 and jsondic['subclass'] is not None:
            mdict['subclass'] = jsondic['subclass']['name']
        if sum([bool(re.match('direct_parent', x)) for x in jsondic.keys()]) > 0 and jsondic['direct_parent'] is not None:
            mdict['direct_parent'] = jsondic['direct_parent']['name']
        if sum([bool(re.match('intermediate_nodes', x)) for x in jsondic.keys()]) > 0 and jsondic['intermediate_nodes'] is not None:
            mdict['intermediate_nodes'] = _get_name_value(jsondic['intermediate_nodes'])
        if sum([bool(re.match('molecular_framework', x)) for x in jsondic.keys()]) > 0 and jsondic['molecular_framework'] is not None:
            mdict['molecular_framework'] = jsondic['molecular_framework']    
    
    return mdict

def make_classy_table_json(jsonresp):  
    
    dmetadatalist = []
    
    if isinstance(jsonresp,dict):
        mdict = _clean_to_table(jsonresp)
        if mdict is not {}:
            dmetadatalist.append(mdict)
    elif isinstance(jsonresp,list):
        for idx,entry in enumerate(jsonresp):
            mdict = _clean_to_table(entry)
            if mdict != {}:
                dmetadatalist.append(mdict)
                
    return(dmetadatalist)

#endregion

#region sdf process

import re

_sdf_marker_pattern = re.compile(r'^> <.*>$')  # match lines starting with > < and ending with >

def _clean_sdf_section(section):
    if not section:
        return []
    # check if the section ends with $$$$
    if section[-1] != '$$$$':
        return section
    end_line = section[-1]
    section_lines = section[:-1]  # remove the last line
    # find all lines that match the marker pattern
    marker_indices = [i for i, line in enumerate(section_lines) if _sdf_marker_pattern.match(line)]
    
    new_section_lines = []
    prev_end = 0  # former the end index of the previous marker
    
    for idx, marker_i in enumerate(marker_indices):
        # add lines between markers to the new section
        new_section_lines.extend(section_lines[prev_end:marker_i])
        current_marker = section_lines[marker_i]
        # check if the current marker is the last one
        if idx + 1 < len(marker_indices):
            next_marker_i = marker_indices[idx + 1]
        else:
            next_marker_i = len(section_lines)
        # get the content between the current marker and the next marker
        content = section_lines[marker_i + 1 : next_marker_i]
        # process the content
        if content:
            if content[-1].strip() != '':
                content.append('')
        else:
            content.append('')  # if content is empty, add an empty line
        # add the current marker and the processed content to the new section
        new_section_lines.append(current_marker)
        new_section_lines.extend(content)
        prev_end = next_marker_i
    
    # add the last part of the section
    new_section_lines.extend(section_lines[prev_end:])
    
    # check if the last line is empty and remove it if it is
    if new_section_lines and new_section_lines[-1].strip() != '':
        new_section_lines.append('')
    new_section_lines.append(end_line)
    
    return new_section_lines

# def _clean_sdf_file(file_path):
#     sections = []
#     current_section = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.rstrip('\n')
#             current_section.append(line)
#             if line == '$$$$':
#                 sections.append(current_section)
#                 current_section = []

#     if current_section:
#         sections.append(current_section)

#     processed_sections = []
#     for section in sections:
#         processed = _clean_sdf_section(section)
#         processed_sections.append(processed)

#     with open(file_path, 'w') as f:
#         for section in processed_sections:
#             for line in section:
#                 f.write(line + '\n')

def _clean_sdf_string(text):
    sections = []
    current_section = []
    for line in text.split('\n'):
        current_section.append(line)
        if line == '$$$$':
            sections.append(current_section)
            current_section = []

    if current_section:
        sections.append(current_section)

    processed_sections = []
    for section in sections:
        processed = _clean_sdf_section(section)
        processed_sections.append(processed)

    sdf_string = '\n'.join(['\n'.join(section) for section in processed_sections])
    return sdf_string

# from io import StringIO
from rdkit import Chem
from io import BytesIO

_result_keys = ['inchikey','smiles',
            'kingdom','superclass',
            'class','subclass',
            'direct_parent','intermediate_nodes','molecular_framework']

def make_classy_table_sdf(sdfresp):
    sdf_string = _clean_sdf_string(sdfresp)
    sdf_bytes = sdf_string.encode('utf-8')  
    sdf_handle = BytesIO(sdf_bytes)  
    mols = Chem.ForwardSDMolSupplier(sdf_handle)

    sdf_dict_list = []
    for mol1 in mols:
        # propNames = list(mol1.GetPropNames())
        dict1 = mol1.GetPropsAsDict()
        new_dict1 = {}
        for k, v in dict1.items():
            new_k = k.lower().replace(' ', '_').strip()
            new_v = v.strip().strip('\n')
            if new_v == '' or new_v == None:
                continue
            if '\t' in new_v:
                new_v = new_v.split('\t')
            new_dict1[new_k] = new_v
        
        if 'inchikey' in new_dict1.keys():
            new_dict1['inchikey'] = new_dict1['inchikey'].split("=")[1]
        
        new_dict2 = {}

        for std_k in _result_keys:
            if std_k == 'intermediate_nodes':
                nodes = []
                if 'intermediate_nodes' in new_dict1.keys():
                    if type(new_dict1['intermediate_nodes']) == list:
                        for v in new_dict1['intermediate_nodes']:
                            nodes.append(v)
                    else:
                        nodes.append(new_dict1['intermediate_nodes'])

                new_dict2[std_k] = nodes
            else:
                if std_k in new_dict1.keys():
                    new_dict2[std_k] = new_dict1[std_k]
            
        sdf_dict_list.append(new_dict2)
    
    return sdf_dict_list

#endregion

#region classyfire query
def structure_query(compound, label='pyclassyfire',**kwargs):
    """
    >>> structure_query('CCC', 'smiles_test')
    >>> structure_query('InChI=1S/C3H4O3/c1-2(4)3(5)6/h1H3,(H,5,6)')

    """

    if ('requests_session' in kwargs.keys()):
        if isinstance(kwargs['requests_session'],requests.sessions.Session):
            s = kwargs['requests_session']
        else: 
            s = requests.session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
            s.mount('http://', HTTPAdapter(max_retries=retries))

    else: 
        s = requests.session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
        s.mount('http://', HTTPAdapter(max_retries=retries))

    try:
        r = s.post(url + '/queries.json', data='{"label": "%s", '
                          '"query_input": "%s", "query_type": "STRUCTURE"}'
                                                      % (label, compound),
                          headers={"Content-Type": "application/json"})
        r.raise_for_status()
        # print(r.json()['id'])
        return  r.json()['id']
    except Exception as e:
        print(f'Error_structure_query:{str(e)}')
        return None

url = "http://classyfire.wishartlab.com"
proxy_url =  "https://gnps-classyfire.ucsd.edu"

def iupac_query(compound, label='pyclassyfire',**kwargs):
    """
    >>> iupac_query('ethane', 'iupac_test')
    >>> iupac_query('C001\\tethane\\nC002\\tethanol', 'iupac_test')
    """
    
    if ('requests_session' in kwargs.keys()):
        if isinstance(kwargs['requests_session'],requests.sessions.Session):
            s = kwargs['requests_session']
        else: 
            s = requests.session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
            s.mount('http://', HTTPAdapter(max_retries=retries))

    else: 
        s = requests.session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
        s.mount('http://', HTTPAdapter(max_retries=retries))
    
    try:
        r = s.post(url + '/queries.json', data='{"label": "%s", '
                          '"query_input": "%s", "query_type": "IUPAC_NAME"}'
                                                      % (label, compound),
                          headers={"Content-Type": "application/json"})
        r.raise_for_status()
        return r.json()['id']
    except:
        return None

def get_results(query_id, return_format="json", blocking=False,**kwargs):
    """
    >>> get_results('595535', 'csv')
    >>> get_results('595535', 'json')
    >>> get_results('595535', 'sdf')

    """

    if query_id is None:
        return None
    
    if ('requests_session' in kwargs.keys()):
        if isinstance(kwargs['requests_session'],requests.sessions.Session):
            s = kwargs['requests_session']
        else: 
            s = requests.session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
            s.mount('http://', HTTPAdapter(max_retries=retries))

    else: 
        s = requests.session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
        s.mount('http://', HTTPAdapter(max_retries=retries))
    
    if blocking == False:
        try:
            r = s.get('%s/queries/%s.%s' % (url, query_id, return_format),
                             headers={"Content-Type": "application/%s" % return_format})
            r.raise_for_status()
            return json.loads(r.text)
        except:
            return None
    else:
        while True:
            try:
                r = s.get('%s/queries/%s.%s' % (url, query_id, return_format),
                                 headers={"Content-Type": "application/%s" % return_format})
                r.raise_for_status()
                result_json = r.json()
                if result_json["classification_status"] != "In Queue":
                    return json.loads(r.text)
                else:
                    print("WAITING")
                    time.sleep(10)
                    continue
            except requests.exceptions.RequestException as e:
                print(f'Error_get_results:{str(e)}')
                return None

from bs4 import BeautifulSoup 

def get_results_json(query_id, return_format="json", query_amount = None,**kwargs):
    
    if query_id is None:
        return None
    
    if ('requests_session' in kwargs.keys()):
        if isinstance(kwargs['requests_session'],requests.sessions.Session):
            s = kwargs['requests_session']
        else: 
            s = requests.session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
            s.mount('http://', HTTPAdapter(max_retries=retries))

    else: 
        s = requests.session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
        s.mount('http://', HTTPAdapter(max_retries=retries))
    
    max_retry_count = 5
    results = []
    
    if query_amount is not None and query_amount < 100:
        sleep_time = query_amount/2 if query_amount/2 > 10 else 10
        retry_count = 0
        while True:
            try:

                r = s.get('%s/queries/%s.%s' % (url, query_id, return_format),
                                    headers={"Content-Type": "application/%s" % return_format,
                                    "Cache-Control": "no-cache",
                                    "Pragma": "no-cache"})
                r.raise_for_status()
                # if r.status_code == 500:
                #     return None
                jsresp = json.loads(r.text)
                if jsresp["classification_status"] == "Done" or retry_count >= max_retry_count: 
                    results = jsresp['entities']
                    break
                else:
                    retry_count += 1
                    time.sleep(sleep_time)

            except requests.exceptions.RequestException:
                print(f"Error on retreive id {query_id}")
                return None
        
    else:
        sleep_time = 10
        r = s.get('%s/queries/%s.%s' % (url, query_id, return_format),
                            headers={"Content-Type": "application/%s" % return_format,
                                    "Cache-Control": "no-cache",
                                    "Pragma": "no-cache"})
        jsresp = json.loads(r.text)
        
        num_pages = jsresp["number_of_pages"]
        respo_dict_list = []
        
        for i in tqdm(range(1, num_pages + 1),leave = False):
            retry_count = 0
            while True:
                try:
                    ind_request = s.get('%s/queries/%s.%s?page=%s' % (url, query_id, return_format, i),
                                        headers={"Content-Type": "application/%s" % return_format})
                    ind_request.raise_for_status()
                    
                    jsresp = json.loads(ind_request.text)
                    if jsresp["classification_status"] == "Done" or retry_count >= max_retry_count:
                        respo_dict_list.append(jsresp)
                        time.sleep(sleep_time)
                        break
                    else:
                        time.sleep(sleep_time)
                        retry_count += 1
                        continue
                        
                except requests.exceptions.RequestException:
                    print(f"Error on retreive id {query_id}")
                    break
        
        results = []
        for j in respo_dict_list:
            results.extend(j['entities'])
    
    results = make_classy_table_json(results)
    return results

def get_results_sdf(query_id, return_format="sdf", query_amount = None,**kwargs):
    
    if query_id is None:
        return None
    
    if ('requests_session' in kwargs.keys()):
        if isinstance(kwargs['requests_session'],requests.sessions.Session):
            s = kwargs['requests_session']
        else: 
            s = requests.session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504, 429])
            s.mount('http://', HTTPAdapter(max_retries=retries))

    else: 
        s = requests.session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
        s.mount('http://', HTTPAdapter(max_retries=retries))
    
    max_try = 3
    try_count = 0
    sleep_time = 10 if query_amount is None else (query_amount / 2 if query_amount / 2 > 10 else 10)

    while try_count < max_try:

        r = s.get('%s/queries/%s' % (url, query_id),
                                headers={"Cache-Control": "no-cache",
                                        "Pragma": "no-cache"})
        if r.status_code == 500:
            return None
        
        soup = BeautifulSoup(r.text, 'html.parser')  
        progress_span = soup.select_one('div.progress-bar > span')  

        if progress_span and progress_span.text.strip() == '100% Complete':  
            break
        else:
            if query_amount is not None:
                row_counts = []
                tables = soup.find_all('table', class_='results table table-striped')
                for idx, table in enumerate(tables, 1):
                    tbody_list = table.find_all('tbody')
                    total_rows = 0
                    for tbody in tbody_list:
                        rows = tbody.find_all('tr')
                        total_rows += len(rows)
                    row_counts.append(total_rows)
            
                if sum(row_counts) == query_amount:
                    break
                else:
                    time.sleep(sleep_time)
                    try_count += 1
            else:
                time.sleep(sleep_time)
                try_count += 1

    r = s.get('%s/queries/%s.%s' % (url, query_id, return_format),
                                headers={"Content-Type": "application/%s" % return_format,
                                        "Cache-Control": "no-cache",
                                        "Pragma": "no-cache"})
    
    results = make_classy_table_sdf(r.text)
    return results

def get_entity(inchikey, return_format="json", gnps_proxy = False,**kwargs):
    """
    >>> get_entity("ATUOYWHBWRKTHZ-UHFFFAOYSA-N", 'csv')
    >>> get_entity("ATUOYWHBWRKTHZ-UHFFFAOYSA-N", 'json')
    >>> get_entity("ATUOYWHBWRKTHZ-UHFFFAOYSA-N", 'sdf')

    """
    # inchikey = inchikey.replace('InChIKey=', '')
    if ('requests_session' in kwargs.keys()):
        if isinstance(kwargs['requests_session'],requests.sessions.Session):
            s = kwargs['requests_session']
        else: s = requests
    else: s = requests
    
    try:
        if gnps_proxy == True:
            r = s.get('%s/entities/%s.%s' % (proxy_url, inchikey, return_format),
                         headers={
                             "Content-Type": "application/%s" % return_format})
            
        else:
            r = s.get('%s/entities/%s.%s' % (url, inchikey, return_format),
                         headers={
                             "Content-Type": "application/%s" % return_format})
        if r.status_code == 404:
            return None
        elif r.status_code == 500:
            return None
        else:
            r.raise_for_status()
        # time.sleep(0.1)
        return json.loads(r.text)
    except Exception as e:
        # print(f'Error_get_entity:{str(e)}')
        return None
    
#endregion   

#region batch query
from tqdm import tqdm

def batch_inchikey_clasy(inchilist, **kwargs):
    '''
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    import requests
    
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
    
    with requests.session() as ses:
        ses.mount('http://', HTTPAdapter(max_retries=retries))
        results = batch_inchikey_clasy(inchilist,requests_session = ses)

    '''
    if ('requests_session' in kwargs.keys()):
        if isinstance(kwargs['requests_session'],requests.sessions.Session):
            s = kwargs['requests_session']

            s_flag = False
        else: 
            s = requests.session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
            s.mount('http://', HTTPAdapter(max_retries=retries))

            s_flag = True
    else: 
        s = requests.session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
        s.mount('http://', HTTPAdapter(max_retries=retries))

        s_flag = True

    all_inchi_keys = [inchi for inchi in inchilist if inchi != '' and inchi is not None and inchi is not np.nan]
    # all_json = run_parallel_job(get_entity, all_inchi_keys, parallelism_level)
    all_json = []

    for inchi in tqdm(all_inchi_keys, leave = False):
        result = get_entity(inchi,requests_session = s)
        if result is not None:
            all_json.append(result)
        time.sleep(0.5)

    results = make_classy_table_json(all_json)

    if s_flag: s.close()
    return results

def _split_list(input_list, batch_size):
    return [input_list[i:i+batch_size] for i in range(0, len(input_list), batch_size)]

def batch_struc_clasy(strulist, batch_size = 40, return_format = 'sdf', **kwargs):
    '''
    when using json to fetch data, if there are invalid entry, it will fail to retrieve the data even if there are valid entries
    use sdf will not have this problem
    
    with requests.session() as ses:
        ses.mount('http://', HTTPAdapter(max_retries=retries))
        results = batch_struc_clasy(strulist,requests_session = ses)

    '''
    if ('requests_session' in kwargs.keys()):
        if isinstance(kwargs['requests_session'],requests.sessions.Session):
            s = kwargs['requests_session']

            s_flag = False
        else: 
            s = requests.session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
            s.mount('http://', HTTPAdapter(max_retries=retries))

            s_flag = True
    else: 
        s = requests.session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504 , 429])
        s.mount('http://', HTTPAdapter(max_retries=retries))

        s_flag = True
    
    if return_format == 'json':
        get_results_func = get_results_json
    elif return_format == 'sdf':
        get_results_func = get_results_sdf
    else:
        raise ValueError('return_format must be "json" or "sdf"')
    
    batch_size = min(100, batch_size)
    batch_list = _split_list(strulist, batch_size)
    max_retry = 3
    error_batch_list = []
    error_ids_list = []
    results = []
    for minibatch in tqdm(batch_list, leave = False):
        query_amount = len(minibatch)
        query_id = structure_query('\\n'.join(minibatch),**kwargs)
        time.sleep(10)
        if query_id != None:
            result0 = get_results_func(query_id, query_amount=query_amount,**kwargs)
            if result0 != None:
                results.extend(result0)
            else:
                error_ids_list.append(query_id)
        else:
            error_batch_list.append(minibatch)
    
    if len(error_batch_list) > 0:
        for minibatch in tqdm(error_batch_list, leave = False):
            retry_count = 0
            while True or retry_count < max_retry:
                query_amount = len(minibatch)
                query_id = structure_query('\\n'.join(minibatch),**kwargs)
                time.sleep(10)
                if query_id != None:
                    result0 = get_results_func(query_id, query_amount=query_amount,**kwargs)
                    results.extend(result0)
                    break
                else:
                    error_ids_list.append(query_id)
                    if retry_count >= max_retry:
                        break
                retry_count += 1
                time.sleep(10) 

    if len(error_ids_list) > 0:
        for query_id in tqdm(error_ids_list, leave = False):
            retry_count = 0
            while True or retry_count < max_retry:
                result0 = get_results_func(query_id, query_amount=query_amount,**kwargs)
                if result0 != None:
                    results.extend(result0)
                    break
                else:
                    if retry_count >= max_retry:
                        break
                retry_count += 1
                time.sleep(10)

    if s_flag: s.close()
    return results

#endregion

#region test
if __name__ == '__main__':
    
# t1 = get_entity("ATUOYWHBWRKTHZ-UHFFFAOYSA-N", 'json')
# t2 = json.loads(t1)
# d1 = make_classy_table(t2)

    inchilist = [
    'AAALVYBICLMAMA-UHFFFAOYSA-N',
    'AAEVYOVXGOFMJO-UHFFFAOYSA-N',
    'AAFNEINEQRQMTF-WUUYCOTASA-N',
    'AAFXQFIGKBLKMC-UHFFFAOYSA-N',
    'AAIBYZBZXNWTPP-UHFFFAOYSA-N',
    'AAKDPDFZMNYDLR-UHFFFAOYSA-N',
    'AAKJLRGGTJKAMG-UHFFFAOYSA-N',
    'AAMHBRRZYSORSH-UHFFFAOYSA-N',
    'AAOVKJBEBIDNHE-UHFFFAOYSA-N',
    'AAPVQEMYVNZIOO-UHFFFAOYSA-N',
    None,
    'AATNZNJRDOVKDD-UHFFFAOYSA-N',
    'AAXVEMMRQDVLJB-BULBTXNYSA-N',
    'ABBQHOQBGMUPJH-UHFFFAOYSA-M',
    'ABCSSKWSUJMJCP-WQDFMEOSSA-N',
    'ABDKAPXRBAPSQN-UHFFFAOYSA-N',
    ]
    
    strulist = [
        'CC1=CC=C(C=C1)S(=O)(=O)C(C)(C)C(=O)C2=CC=C(C=C2)SC3=CC=C(C=C3)C(=O)C4=CC=CC=C4',
        'CCCCCCCCCCCCCCCCS(=O)(=O)O',
        'CCCCCCCCCCCCCCCS(=O)(=O)O',
        'CCCCCCCCCCCCCCS(=O)(=O)O',
        'C1=CC=C2C=C(C=CC2=C1)S(=O)(=O)O',
        'C(CO)N(CCO)CCO',
        'C1=CC(=CC=C1CC2=CC=C(C=C2)O)O',
        'CC(=CCC/C(=C/CC/C(=C/CC/C=C(/CC/C=C(/CCC=C(C)C)\C)\C)/C)/C)C',
        'CCCCCCCCCCCCCCCC(=O)OCCCC',
        'CCCCC(CC)COC(=O)/C=C/C(=O)OCC(CC)CCCC',
        'CCCCCCCCCCCC(=O)O',
        'CC(C)(C)C1=CC(=CC=C1)C(C)(C)C',
        'CCCCC(CC)CO',
        'CCCCCCCCCCCCCCCC(=O)OC',
        'CCCCCCCC/C=C\CCCCCCCC(=O)OC',
        'CCCCCCCC/C=C\CCCCCCCC(=O)O',
        'C1=CC=C(C=C1)C(=O)C2=CC=CC=C2',
        'CC(C)C1=CC2=C(C=C1)[C@]3(CCC[C@@]([C@@H]3CC2)(C)C(=O)OC)C',
        'CCCCCCCCCCCCCCCCCC(=O)OCCCC',
        'CCCCC(CC)COP(=O)(OC1=CC=CC=C1)OC2=CC=CC=C2',
        'CC(C)C1=CC2=CC[C@@H]3[C@@]([C@H]2CC1)(CCC[C@@]3(C)C(=O)OC)C',
        'CC1=CC(=C(C(=C1)C(C)(C)C)O)C(C)(C)C',
        'CC(C)C1=CC2=C(C=C1)[C@]3(CCC[C@@]([C@@H]3CC2)(C)C=O)C',
        'CC(C)C1=CC2=C(C=C1)[C@]3(CCC[C@@]([C@@H]3CC2)(C)C(=O)O)C',
        'CC(C)C1CCC2=C(C1)CCC3C2(CCCC3(C)C(=O)O)C',
        'C1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)C3=CC=CC=C3',
        'CC(C)C1=CC2=C(C=C1)C=C(C=C2)C(C)C',
        'CC(C)(C)C1=CC(=C)C=C(C1=O)C(C)(C)C',
    ]
    
    # with requests.session() as ses:
    #     ses.mount('http://', HTTPAdapter(max_retries=retries))
    #     results = batch_inchikey_clasy(inchilist,parallelism_level = 2,requests_session = ses)
    
    results_json = batch_struc_clasy(strulist, return_format='json')
    # results1 = batch_inchikey_clasy(inchilist)
    results_sdf = batch_struc_clasy(strulist, return_format='sdf')
    import pandas as pd
    results_json = pd.DataFrame(results_json)
    results_sdf = pd.DataFrame(results_sdf)

    results_json.equals(results_sdf)

    smi_list = ['CCCCCCCCCCCCCCCCCC(=O)O[PbH2]OC(=O)CCCCCCCCCCCCCCCCC',
        'N#CS(=O)(=O)c1ccccc1',
        'CCCCC(CC)C(=O)O[SnH2]OC(=O)C(CC)CCCC',
        'N#CSC=CSC#N',
        'NCCO.O=S(=O)([O-])c1cc(Nc2nc(Nc3ccccc3)nc(N(CCO)CCO)n2)ccc1C=Cc1ccc(Nc2nc(Nc3ccccc3)nc(N(CCO)CCO)n2)cc1S(=O)(=O)[O-].[K+].[K+]',
        'CCCCCCCC[Sn+2]CCCCCCCC.O=C([O-])C=CC(=O)[O-]',
        'O=S(=O)(O)c1ccc(S(=O)(=O)O)c(Nc2nc(Nc3ccc(C=Cc4ccc(Nc5nc(Nc6cc(S(=O)(=O)O)ccc6S(=O)(=O)O)nc(N6CCOCC6)n5)cc4S(=O)(=O)O)c(S(=O)(=O)O)c3)nc(N3CCOCC3)n2)c1.[Na+]',
        '[Al+3].[Al+3].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[Si+4].[Si+4].[Si+4]',
        'C.C.CCCCCCCCC=CCCCCCCCC(=O)OCC[S-].CCCCCCCCC=CCCCCCCCC(=O)OCC[S-].CCCCCCCCC=CCCCCCCCC(=O)OCC[S-].CCCCCCCCC=CCCCCCCCC(=O)OCC[S-].S.[Sn+2].[Sn+2]',
        '[Ca+2].[O-][Br+2]([O-])[O-].[O-][Br+2]([O-])[O-]',
        'CCCCOC(=O)C=CC(=O)O[Sn](CCCC)(CCCC)OC(=O)C=CC(=O)OCCCC',
        'CNC.O=C(C=C(O)C(F)(F)F)c1ccccc1.O=C(C=C(O)C(F)(F)F)c1ccccc1.O=C(C=C(O)C(F)(F)F)c1ccccc1.O=C(C=C(O)C(F)(F)F)c1ccccc1.[Eu].[H+]',
        'O=C([CH-]C(=O)C(F)(F)F)c1cccs1.O=C([CH-]C(=O)C(F)(F)F)c1cccs1.O=C([CH-]C(=O)C(F)(F)F)c1cccs1.O=P(c1ccccc1)(c1ccccc1)c1ccccc1.O=P(c1ccccc1)(c1ccccc1)c1ccccc1.[Eu+3]',
        'CCCCCCC([O-])CC=CCCCCCCCC(=O)O.[Li+]',
        'CC[NH+]=C1C=CC(=C(c2ccc(N(CC)CC)cc2)c2ccc(N(CC)CC)cc2)c2ccccc21.CC[NH+]=C1C=CC(=C(c2ccc(N(CC)CC)cc2)c2ccc(N(CC)CC)cc2)c2ccccc21.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[Cu+].[Cu+].[Fe+2]',
        'CC(C)c1ccc2c(c1)CCC1C(C)(C(=O)[O-])CCCC21C.[K+]',
        'CCCCCCCCCc1ccc(OCCOCCCS(=O)(=O)[O-])cc1',
        'C1=CCCC=CCC1.Cl[Rh]Cl',
        '[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W]',
        'CC1=C2N=C(C=C3NC(=C(C)C4=NC(C)(C5N=C1C(C)(CCC(=O)NCC(C)OP(=O)(O)OC1C(CO)OC([n+]6c[nH]c7cc(C)c(C)cc76)C1O)C5CC(N)=O)C(C)(CC(N)=O)C4CCC(=N)[O-])C(C)(CC(N)=O)C3CCC(=N)[O-])C(C)(C)C2CCC(=N)[O-].[C]#N.[Co+2]']

    results1 = batch_struc_clasy(smi_list, return_format = "sdf",)

#endregion