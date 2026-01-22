#region classyfire fetcher
import pyclassfire2 as pcf
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem
import numpy as np
import re
from rdkit.Chem import PandasTools
import sys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_base_dir():
    
    # script_path = Path(__file__).resolve() # use when this file is in main folder
    # return script_path.parent

    current_file_dir = os.path.dirname(__file__)
    main_dir = os.path.dirname(current_file_dir)
    refinfo_dir = os.path.join(main_dir, 'refinfo')

    if not os.path.exists(refinfo_dir):
        raise FileNotFoundError(f"refinfo directory not found at {refinfo_dir}")
    
    return main_dir

_classyfire_cache_file = 'refinfo/classyfire/clf_cache.feather'
_classyfire_cache_file = os.path.join(get_base_dir(), _classyfire_cache_file)

_food_list_file = "refinfo/foodb/Food.csv"
_food_list_file = os.path.join(get_base_dir(), _food_list_file)

_pubchem_cache_folder = 'refinfo/pubchem'
_pubchem_cache_folder = os.path.join(get_base_dir(), _pubchem_cache_folder)

_pubchem_cache_file = 'refinfo/pubchem/pbc_cache.feather'
_pubchem_cache_file = os.path.join(get_base_dir(), _pubchem_cache_file)

_comptox_cache_folder = 'refinfo/comptox'
_comptox_cache_folder = os.path.join(get_base_dir(), _comptox_cache_folder)

_comptox_cache_file = 'refinfo/comptox/cpt_cache.feather'
_comptox_cache_file = os.path.join(get_base_dir(), _comptox_cache_file)

_nist_cache_file = 'refinfo/nist20/nist20.feather'
_nist_cache_file = os.path.join(get_base_dir(), _nist_cache_file)

def split_list(input_list, batch_size):
    return [input_list[i:i+batch_size] for i in range(0, len(input_list), batch_size)]

_ichikey_pattern = r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$'

def _check_ik(string, pattern=_ichikey_pattern):

    if string in ['', np.NaN, None]:
        return False

    if not isinstance(string, str):
        string = str(string)
        # if np.isnan(string):
        #     string = ""
    if re.match(pattern, string):
        return True
    else:
        return False

def _std_inchikey(inchikey):  
    if pd.isna(inchikey):
        return None  
    idx = inchikey.find('-') 
    if idx == -1: 
        return inchikey + "-UHFFFAOYSA-N"  
    return inchikey[:idx] + "-UHFFFAOYSA-N" 

def _clean_inchikey(inchikey, stdize=False):  
    if pd.isna(inchikey):
        return None  
    if _check_ik(inchikey):
        if stdize:
            return _std_inchikey(inchikey)
        else:
            return inchikey
    else:
        return None

def _clean_smi(smiles):
    if smiles in ['', np.NaN, None]:
        return None
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            smiles = Chem.MolToSmiles(mol)
            return smiles
        else:
            return None

_clf_df_col = ['inchikey', 'smiles', 'kingdom', 'superclass', 'class', 'subclass',
            'direct_parent', 'intermediate_nodes', 'molecular_framework']

def _clean_input_to_df(input_list, mode = 'inchikey'):
    query_df = pd.DataFrame({'input':input_list})
    if mode == 'inchikey':
        # query_df['stdinchikey'] = query_df['query'].map(stdize_inchikey)
        query_df['inchikey'] = query_df['input'].map(_clean_inchikey)

    elif mode == 'smiles':
        # PandasTools.AddMoleculeColumnToFrame(query_df, smilesCol='input', molCol='ROMol',)
        query_df['querysmiles'] = query_df['input'].map(lambda x: _clean_smi(x))
        query_df['inchikey'] = query_df['querysmiles'].map(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)) if x else None)
        
        # query_df['stdinchikey'] = query_df['inchikey'].map(stdize_inchikey)

    query_df['inchikey'] = query_df['inchikey'].map(_clean_inchikey)
    query_df = query_df.dropna(how='any').drop_duplicates(subset=['inchikey'])
    
    return query_df

class Classyfire_Fetcher:
    def __init__(self, cache_file = _classyfire_cache_file):
        self.cache_file = cache_file
        if not os.path.exists(cache_file):
            # os.makedirs(cache_path)
            print(f'Build new cache')
            clf_df = pd.DataFrame(columns=_clf_df_col)
        else:
            print(f'Load cache from {cache_file}')
            clf_df = pd.read_feather(cache_file)
            assert clf_df.columns.to_list() == _clf_df_col, 'cache file format error'
            
        # clf_df['stdinchikey'] = clf_df['inchikey'].map(stdize_inchikey)
        self.batchsize = 40
        self.cache = clf_df
    
    def _append_to_cache(self, update_dict, col=_clf_df_col):
        if len(update_dict) == 0:
            return
        update_df = pd.DataFrame(update_dict)[col]
        self.cache = pd.concat([self.cache, update_df], ignore_index=True)
        self.cache.drop_duplicates(subset=['inchikey'], inplace=True, ignore_index=True)
        self.cache.dropna(subset=['inchikey'], inplace=True)
        self.cache.to_feather(self.cache_file)
        print(f'Cache updated with {len(update_dict)} new records')

    def _save_cache(self, df):
        assert df.columns.tolist() == _clf_df_col, 'cache file format error'
        df.to_feather(self.cache_file)
        print(f'Cache saved')

    def _inchikey_clasy(self, inchikey_list):

        # batch_list = split_list(inchikey_list, batch_size=self.batchsize)
        # searched_info = []
        # dict_list = []
        # for batlist in tqdm(batch_list, position=0, desc=f'Querying classyfire with {len(batch_list)} batch', leave=False):
        #     dict_list = pcf.batch_inchikey_clasy(batlist)
        #     searched_info.extend(dict_list)
        # searched_info = [dict1 for dict1 in searched_info if dict1 is not None]

        if inchikey_list is None or len(inchikey_list) == 0:
            return []

        dict_list = pcf.batch_inchikey_clasy(inchikey_list)
        dict_list = [dict1 for dict1 in dict_list if dict1 is not None]

        return dict_list

    def _smiles_clasy(self, smiles_list):

        # batch_list = split_list(smiles_list, batch_size=self.batchsize)
        # searched_info = []
        # dict_list = []
        # for batlist in tqdm(batch_list, position=0, desc=f'Querying classyfire with {len(batch_list)} batch', leave=False):
        #     dict_list = pcf.batch_struc_clasy(batlist)
        #     searched_info.extend(dict_list)
        # searched_info = [dict1 for dict1 in searched_info if dict1 is not None]

        if smiles_list is None or len(smiles_list) == 0:
            return []

        dict_list = pcf.batch_struc_clasy(smiles_list)
        dict_list = [dict1 for dict1 in dict_list if dict1 is not None]
        
        return dict_list

    def fetch_classfire(self, query_list, mode = 'inchikey', use_cache = True, use_service = True, update_cache = True, update_interval = 2):
        assert mode in ['inchikey', 'smiles'], 'mode should be inchikey or smiles'

        query_df = _clean_input_to_df(query_list, mode=mode)
        query_iky_list = query_df['inchikey'].values.tolist()

        cache = []
        if use_cache:
            cache = self.cache[self.cache['inchikey'].isin(query_iky_list)]
            query_iky_list = [ik for ik in query_iky_list if ik not in cache['inchikey'].values.tolist()]
            cache = cache.to_dict(orient='records')

        if use_service and len(query_iky_list) > 0:
            # print(F'Querying with {len(query_iky_list)} inchikeys')
            
            batch_list = split_list(query_iky_list, batch_size=self.batchsize)

            update_count = 0
            dict_list = []

            for blist in tqdm(batch_list, position=0, desc=f'Querying classyfire with {len(batch_list)} batch', leave=False):

                dict_list.extend(self._inchikey_clasy(blist))
                # print(f'use_inchi_get_len: {len(dict_list)}',end=' ')
                # print(dict_list)
                if mode == 'smiles':
                    
                    results0 = pd.DataFrame(dict_list)
                    if len(results0) > 0 and 'inchikey' in results0.columns:
                        done_list = results0['inchikey'].values
                    else:
                        done_list = []
                    leftout_smi_list = query_df[(query_df['inchikey'].isin(blist)) & (~query_df['inchikey'].isin(done_list))]['querysmiles'].values.tolist()
                    # print(F'Querying with {len(leftout_smi_list)} smiles')
                    if len(leftout_smi_list) > 0:
                        dict_list.extend(self._smiles_clasy(leftout_smi_list))
                        # print(f'use_smi_get_len: {len(dict_list)}',end=' ')
                
                cache.extend(dict_list)
                
                update_count += 1
                if update_cache and update_count % update_interval == 0:
                    self._append_to_cache(dict_list)
                    dict_list = []
                
        if len(cache) > 0:
            cache = pd.DataFrame(cache)
            query_df = pd.merge(query_df, cache, on='inchikey', how='left')
        else:
            print(F'No found for {len(query_iky_list)} entries')

        return query_df

#endregion

#region fetch pubchem
import deps.pubchempy2 as pcp
import pandas as pd
import numpy as np
import re

_compound_attr = ['cid', 'iupac_name', 'synonyms', 'molecular_formula',
       'molecular_weight', 'exact_mass', 'canonical_smiles', 'isomeric_smiles',
       'inchi', 'inchikey', 'fingerprint', 'cactvs_fingerprint', 'charge',
       'xlogp', 'tpsa', 'h_bond_acceptor_count', 'h_bond_donor_count',
       'heavy_atom_count',
        #  'casrn', 'dtxsid'
                  ]

_pbc_col_list = ['cid', 'iupac_name', 'synonyms', 'molecular_formula',
       'molecular_weight', 'exact_mass', 'canonical_smiles', 'isomeric_smiles',
       'inchi', 'inchikey', 'fingerprint', 'cactvs_fingerprint', 'charge',
       'xlogp', 'tpsa', 'h_bond_acceptor_count', 'h_bond_donor_count', 'heavy_atom_count',
         'casrn', 'dtxsid'
                  ]

_cas_pattern = re.compile(r'\b\d{1,9}-\d{2}-\d\b')
_dtxsid_pattern = re.compile(r'DTXSID\d+')

def _extract_pattern_str(string_list, pattern):
    matched = []
    for string in string_list:
        if re.match(pattern, string):
            matched.append(string)
    if len(matched) == 0:
        matched = None
    return matched

def _clean_liststr(list_str):
    '''
    pbc_f = PubChem_Fetcher()
    t1 = pbc_f.cache
    t1['casrn'] = t1['casrn'].map(_clean_liststr)
    t1['dtxsid'] = t1['dtxsid'].map(_clean_liststr)
    '''
    if list_str is not None:
        if '[' in list_str:
            list_str = eval(list_str)
            if isinstance(list_str, list):
                return list_str
        else:
            list_str = [list_str]
        return list_str
    else:
        return None

def _get_list_first(list1):  
    if list1 is None:  
        return None  
    elif isinstance(list1, list) or isinstance(list1, np.ndarray):  
        return list1[0]  
    else:
        return list1

def _synonyms_extract(info_dict, get_cas=True, get_dtxsid=True):
    if not "synonyms" in info_dict.keys():
        return info_dict

    if not isinstance(info_dict["synonyms"], list):
        info_dict["synonyms"] = info_dict["synonyms"]
    syn_list = info_dict["synonyms"]

    info_dict["casrn"] = None
    info_dict["dtxsid"] = None

    if get_cas:
        matched_list = _extract_pattern_str(syn_list, _cas_pattern)
        # if len(matched_list) == 1:
        #     info_dict["casrn"] = matched_list[0]
        # elif len(matched_list) > 1:
        #     info_dict["casrn"] = str(matched_list)
        info_dict["casrn"] = matched_list

    if get_dtxsid:
        matched_list = _extract_pattern_str(syn_list, _dtxsid_pattern)
        # if len(matched_list) == 1:
        #     info_dict["dtxsid"] = matched_list[0]
        # elif len(matched_list) > 1:
        #     info_dict["dtxsid"] = str(matched_list)
        info_dict["dtxsid"] = matched_list

    return info_dict

# _food_list = pd.read_csv(_food_list_file)['name'].unique().tolist()
# _food_list_sci = pd.read_csv(_food_list_file)['name_scientific'].unique().tolist()

_food_list = pd.read_csv(_food_list_file)['name'].tolist()
_food_list_sci = pd.read_csv(_food_list_file)['name_scientific'].tolist()

def _replace_scif_in_food(food_list, food_list_sci, check_list): 
    result = check_list.copy()  

    for i in range(len(result)):  
        if result[i] in food_list_sci:  
            index = food_list_sci.index(result[i])
            result[i] = food_list[index] 

    result = list(set(result))

    return result 

_skip_words = ['jecfa functional classes','jecfa flavorings index','oecd category']
_replace_words = {'agents':'agent',
                    'ingredients':'ingredient',
                    'flavouring':'flavoring'
                    }

def _replace_word(input_str, replace_dict = _replace_words):  
    for key in replace_dict:  
        if key in input_str:
            input_str = input_str.replace(key, replace_dict[key])  
    return input_str  

def _cleanandlower_list(list1):
    if list1 is None:
        return None

    result = []

    for i in range(len(list1)):
        i_str = list1[i].lower().strip()

        if 'hazard classes and categories' in i_str:
            continue

        if ';' in i_str:
            i_sp = i_str.split(';')
            i_str = i_sp[0].strip()

        if '->' in i_str:
            i_sp = i_str.split('->')
            i_sp = [j.strip() for j in i_sp]
            if len(i_sp) == 2:
                i_sp_0, i_sp_1 = i_sp  
    
                if i_sp_1 == "":  
                    i_str = i_sp_0  
                else:  
                    i_str = i_sp_1  

                if i_str in _skip_words:  
                    i_str = i_sp_0 if i_str == i_sp_1 else i_sp_1  

        if i_str == "":
            continue

        i_str = _replace_word(i_str)

        i_str = i_str.replace('_', ' ')
        result.append(i_str)
        
    return result

from typing import Union, List

def find_rows_with_value(df: pd.DataFrame, column_names: Union[str, List[str]], value: object, mode: str = 'or') -> pd.DataFrame:

    if isinstance(column_names, str):
        column_names = [column_names]
    
    if mode not in ['and', 'or']:
        raise ValueError("mode invalid")
    
    def check_value(x):
        return value in x if x is not None else False
    
    masks = []
    for col in column_names:
        masks.append(df[col].apply(check_value))
    
    combined_mask = pd.concat(masks, axis=1)
    
    if mode == 'and':
        final_mask = combined_mask.all(axis=1)
    else:
        final_mask = combined_mask.any(axis=1)
    
    return df[final_mask]


class PubChem_Fetcher:
    def __init__(self, cache_folder = _pubchem_cache_folder, cache_file = _pubchem_cache_file):
        self.cache_file = cache_file
        self.cache_folder = cache_folder

        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        if not os.path.exists(cache_file):
            print(f'Build new cache')
            pbc_df = pd.DataFrame(columns=_compound_attr)
        else:
            print(f'Load cache from {cache_file}')
            pbc_df = pd.read_feather(cache_file)
            assert all(col in pbc_df.columns for col in _compound_attr)  , 'cache file format error'
        
        self.batchsize = 50
        self.cache = pbc_df

    def _append_to_cache(self, update_dict, col = _pbc_col_list):
        if len(update_dict) == 0:
            return
        update_df = pd.DataFrame(update_dict)[col]
        self.cache = pd.concat([self.cache, update_df], ignore_index=True)
        self.cache.drop_duplicates(subset=['cid'], inplace=True, ignore_index=True)
        self.cache.dropna(subset=['cid'], inplace=True)
        self.cache.to_feather(self.cache_file)
        print(f'Cache updated with {len(update_dict)} new records')

    def _save_cache(self,df):
        assert df.columns.tolist() == _pbc_col_list, 'cache file format error'
        df.to_feather(self.cache_file)
        print(f'Cache saved')

    @ staticmethod
    def _fetch_pubchem_basic_one(query_val, namespace, requests_session):
        '''
        with requests.session() as sess:
            info1 = _fetch_pubchem_basic_one('408530-29-0','name',requests_session=sess)
        '''

        assert namespace in ['inchikey', 'name', 'smiles', 'cid']
        info0 = pcp.get_compounds(
            query_val, namespace, requests_session=requests_session)

        if len(info0) != 0:
            return info0[0]
        else:
            return None
        
    def fetch_pubchem_basic(self, query_list, mode = 'inchikey', use_cache = True, use_service = True, update_cache = True, update_interval = 100, sess_size = 50):
        '''
        mode available: inchikey, cas, name, smiles, cid

        return df columns = ['input', 'inchikey', 'cid', 'iupac_name', 'synonyms',
       'molecular_formula', 'molecular_weight', 'exact_mass',
       'canonical_smiles', 'isomeric_smiles', 'inchi', 'fingerprint',
       'cactvs_fingerprint', 'charge', 'xlogp', 'tpsa',
       'h_bond_acceptor_count', 'h_bond_donor_count', 'heavy_atom_count',
       'casrn', 'dtxsid']
        '''
        assert mode in ['inchikey', 'cas', 'name', 'smiles', 'cid']

        if mode in ['inchikey', 'smiles']:
            query_df = _clean_input_to_df(query_list, mode=mode)
            mode = 'inchikey'
            namespace = 'inchikey'
        elif mode == 'cas':
            namespace = 'name'
            query_df = pd.DataFrame({'input':query_list,'cas':query_list})
        elif mode == 'cid':
            cid_list = [int(str(cid)) for cid in query_list]
            query_df = pd.DataFrame({'input':query_list, mode: cid_list})
            namespace = 'cid'
        elif mode == 'name':
            namespace = 'name'
            query_df = pd.DataFrame({'input':query_list,'name':query_list})

        query_df = query_df.dropna(how='any')

        query_val_list = query_df[mode].unique().tolist()
        cache = []
        if use_cache:
            if mode in ['inchikey', 'cid']:
                cache = self.cache[self.cache[mode].isin(query_val_list)]
                query_val_list = [val for val in query_val_list if val not in cache[mode].values.tolist()]
                cache = cache.to_dict(orient='records')
            elif mode == 'cas':
                query_val_list_temp = []
                for val in tqdm(query_val_list, desc='Searching cache:', leave=False):
                    match = find_rows_with_value(self.cache, 'casrn', val)
                    if len(match) > 0:
                        match = match.to_dict(orient='records')[0]
                        match[mode] = val
                        cache.append(match)
                    else:
                        query_val_list_temp.append(val)
                query_val_list = query_val_list_temp
            elif mode == 'name':
                query_val_list_temp = []
                for val in tqdm(query_val_list, desc='Searching cache:', leave=False):
                    match = find_rows_with_value(self.cache, ['iupac_name','synonyms'], val)
                    if len(match) > 0:
                        match = match.to_dict(orient='records')[0]
                        match[mode] = val
                        cache.append(match)
                    else:
                        query_val_list_temp.append(val)
                query_val_list = query_val_list_temp

        if use_service and len(query_val_list) > 0:
            dict_list = []
            with requests.session() as sess:
                sess_count = 0
                for val in tqdm(query_val_list, desc=f'Querying pubchem with {len(query_val_list)} entries', leave=False):
                    try:
                        x = self._fetch_pubchem_basic_one(val, namespace, requests_session=sess)
                    except Exception as ex:
                        print(ex, end="")
                        x = None
                    if x is None:
                        continue
                    else:
                        info0 = x.to_dict(_compound_attr)
                        info0 = _synonyms_extract(info0)
                        if mode not in ['inchikey', 'cid']:
                            info0[mode] = val
                        dict_list.append(info0)

                    sess_count += 1
                    if sess_count >= sess_size:
                        sess_count -= sess_size
                        tqdm.write("reset internet")
                        sess.close()
                        sess = requests.session()

                        # middle save, fail save
                        if update_cache:
                            self._append_to_cache(dict_list)

            if update_cache:
                self._append_to_cache(dict_list)

            cache.extend(dict_list)

        if len(cache) > 0:
            cache = pd.DataFrame(cache)
            query_df = pd.merge(query_df, cache, on=mode, how='left')
        else:                   
            print(F'No found for {len(query_val_list)} entries')

        return query_df
    
    @ staticmethod
    def _fetch_pubchem_taxonomy_one(cid, food_compound_only=True, cache_folder = 'refinfo/pubchem'):  

        if os.path.exists(f'{cache_folder}/{cid}_consolidatedcompoundtaxonomy.csv'):
            df = pd.read_csv(f'{cache_folder}/{cid}_consolidatedcompoundtaxonomy.csv')
        else:
            base_url = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi"  
            query = {  
                "download": "*",  
                "collection": "consolidatedcompoundtaxonomy",  
                "order": ["cid,asc"],  
                "start": 1,  
                "limit": 10000000,  
                "downloadfilename": f"{cid}_consolidatedcompoundtaxonomy",  
                "where": {  
                    "ands": [  
                        {"cid": str(cid)},
                    ]  
                }  
            }  

            params = {  
                "infmt": "json",  
                "outfmt": "csv",  
                "query": json.dumps(query)
            }  

            response = requests.get(base_url, params=params)  
            
            if response.status_code == 200:  
                data = response.text
                csv_data = StringIO(data) 
                df = pd.read_csv(csv_data)
                df.to_csv(f'{cache_folder}/{cid}_consolidatedcompoundtaxonomy.csv', index=False)
                
            else:  
                print(f"Request failed with status code: {response.status_code}")  
                return None

        if food_compound_only:
            df = df[df['srccmpdkind'] == 'Food Compound']
        
        return df['srcname'].unique().tolist()


    @ staticmethod
    def _fetch_pubchem_cpdat_one(cid, cache_folder = 'refinfo/pubchem', oecd_only = True):  

        if os.path.exists(f'{cache_folder}/{cid}_cpdat.csv'):
            df = pd.read_csv(f'{cache_folder}/{cid}_cpdat.csv')
            
        else:
            base_url = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi"  
            query = {  
                "download": "*",  
                "collection": "cpdat",  
                "order": ["category,asc"],  
                "start": 1,  
                "limit": 10000000,  
                "downloadfilename": f"{cid}_cpdat",  
                "where": {  
                    "ands": [  
                        {"cid": str(cid)},
                    ]  
                }  
            }  

            params = {  
                "infmt": "json",  
                "outfmt": "csv",  
                "query": json.dumps(query)
            }  

            response = requests.get(base_url, params=params)  
            
            if response.status_code == 200:  
                data = response.text
                csv_data = StringIO(data) 
                df = pd.read_csv(csv_data)
                df.to_csv(f'{cache_folder}/{cid}_cpdat.csv', index=False)
                
            else:  
                print(f"Request failed with status code: {response.status_code}")  
                return None

        if oecd_only:
            df = df[df['source'] == 'OECD Functional Use']
            if df.empty:
                return []
        
        result = df['category'].unique().tolist()

        return result

    @ staticmethod
    def _fetch_pubchem_classification_one(cid, hid = 101, cache_folder = 'refinfo/pubchem'):
        if os.path.exists(f'{cache_folder}/{cid}_{hid}_classification.json'):
            data = json.load(open(f'{cache_folder}/{cid}_{hid}_classification.json'))
            
            
        else:
            url = "https://pubchem.ncbi.nlm.nih.gov/classification/cgi/classifications.fcgi"  
            params = {  
                "hid": hid,  
                "start": "root",  
                "format": "json",  
                "search_uid_type": "compound",  
                "search_uid": cid,  
                "search_max": 10,  
                "search_type": "list"  
            }  

            response = requests.get(url, params=params)  
            if response.status_code == 200:  
                data = response.text
                data = json.loads(data)
                with open(f'{cache_folder}/{cid}_{hid}_classification.json', 'w') as f:
                    json.dump(data, f)
            else:
                print(f"Request failed with status code: {response.status_code}")
                return None

        node_list = []
        for node0 in data['Hierarchies']['Hierarchy']:
            for node in node0['Node']:
                if node['ParentID'] == ["root"]:
                    data = node['Information']['Name']
                    if isinstance(data, str):
                        node_list.append(data.strip())
                    elif isinstance(data, dict):
                        if 'StringWithMarkup' in data.keys():
                            if 'String' in data['StringWithMarkup'].keys():
                                node_list.append(data['StringWithMarkup']['String'].strip())

        node_list = list(set(node_list))
        return node_list
            
    def fetch_pubchem_json(self, query_list, mode = 'inchikey', food_compound_only=True, folder_path = None):
        
        assert mode in ['inchikey','cid']

        if folder_path is None:
            folder_path = self.cache_folder

        if mode == 'cid':
            query_df = pd.DataFrame({'input': query_list, 'cid': query_list})
        else:
            query_df = pd.DataFrame({'input': query_list, 'inchikey': query_list})
            query_df = pd.merge(query_df, self.cache[['cid','inchikey']], on='inchikey', how='left')
            query_ick_list = query_df[query_df['cid'].isnull()]['inchikey']
            result0 = self.fetch_pubchem_basic(query_ick_list, mode = 'inchikey', use_cache = True, use_service = False, update_cache = True)[['inchikey','cid']]
            query_df = pd.merge(query_df, result0, on='inchikey', how='left')
        
        query_df = query_df.dropna(how = 'any')
        query_df['cid'] = query_df['cid'].astype(int)
        query_df['cid'] = query_df['cid'].astype(str)
        cid_list = query_df['cid'].tolist()

        results = []
        
        # with tqdm(total = len(cid_list), desc = f'Fetching {len(cid_list)} PubChem JSON:') as pbar:
        for cid in tqdm(cid_list, desc = f'Fetching {len(cid_list)} PubChem JSON',leave = False):
            # cid = str(int(cid))
            if not os.path.exists(f"{folder_path}/{cid}.json"):
                
                # pbar.desc = (f"Fetching {cid}")
                req = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON')

                if req.status_code != 200:
                    continue
                
                data = json.loads(req.text)

                with open(f"{folder_path}/{cid}.json","w") as f:
                    json.dump(data,f)

            with open(f"{folder_path}/{cid}.json", 'r') as file:  
                data = json.load(file)  

            result = {}  
            for item in data['Record']['Section']:  
                if item.get('TOCHeading') == 'Names and Identifiers':  
                    
                    for item2 in item['Section']:  
                        if item2.get('TOCHeading') == 'Record Description': 
                            # description = []
                            for item3 in item2['Information']:
                                if item3.get('Description') == 'Ontology Summary':
                                    result['Ontology_Summary'] = item3['Value']['StringWithMarkup'][0]['String']
                                
                if item.get('TOCHeading') == 'Food Additives and Ingredients':  
                    
                    for item2 in item['Section']:
                        
                        if item2.get('TOCHeading') == 'Associated Foods':
                            food_item = self._fetch_pubchem_taxonomy_one(cid, food_compound_only, folder_path)
                            food_item = _replace_scif_in_food(_food_list, _food_list_sci, food_item)
                            result['Associated_Foods'] = food_item

                        if item2.get('TOCHeading') == 'Food Additive Classes':
                            foodaddclass = []
                            for item3 in item2['Information']:
                                foodaddclass.append(item3['Value']['StringWithMarkup'][0]['String'])
                            foodaddclass = _cleanandlower_list(foodaddclass)
                            result['Food_Additive_Classes'] = list(set(foodaddclass))

                        if item2.get('TOCHeading') == 'FDA Substances Added to Food':
                            fda = []
                            for item3 in item2['Information']:
                                if item3.get('Name') == "Used for (Technical Effect)":
                                    fda.append(item3['Value']['StringWithMarkup'][0]['String'])
                            fda = _cleanandlower_list(fda)
                            result['FDA_Substances_Added_to_Food'] = list(set(fda))

                        if item2.get('TOCHeading') == 'Evaluations of the Joint FAO/WHO Expert Committee on Food Additives - JECFA':
                            for item3 in item2['Information']:
                                if item3.get('Name') == 'ADI':
                                    result['Food_Additive_ADI'] = item3['Value']['StringWithMarkup'][0]['String']

                if item.get('TOCHeading') == 'Use and Manufacturing':

                    _temp_tochead = [
                                'Methods of Manufacturing', # see cid = 243 for general section
                                # 'IFRA Fragrance Standards',
                                # 'Sampling Procedures',
                                # 'Impurities',
                                # 'U.S. Exports',
                                # 'U.S. Imports',
                                # 'U.S. Production',
                                # 'Formulations/Preparations',
                                # 'Consumption Patterns',
                                # 'General Manufacturing Information' # see cid=638024 for a natural product having this section
                                ]

                    for item2 in item['Section']:
                        if item2.get('TOCHeading') == 'Uses':

                            if 'Information' in item2:
                                for item3 in item2['Information']:
                                    if item3.get('Name') == 'Sources/Uses':
                                        result.update({'Sources/Uses':item3['Value']['StringWithMarkup'][0]['String']})

                                    if item3.get('Name') == 'Cosmetic Ingredient Review Link':
                                        result.update({'Cosmetic_Ingredient':item3['Value']['StringWithMarkup'][0]['String']})

                                    if item3.get('Name') == 'EPA CPDat Chemical and Product Categories':
                                        cpdat = self._fetch_pubchem_cpdat_one(cid, folder_path)
                                        if len(cpdat) > 0:
                                            result['EPA_CPDat_Chemical_and_Product_Categories'] = cpdat

                            if 'Section' in item2:
                                for item3 in item2['Section']:
                                    if item3.get('TOCHeading') == 'Use Classification':
                                        usemanu = []
                                        for item4 in item3['Information']:
                                            item4str = item4['Value']['StringWithMarkup'][0]['String']
                                            if 'Safer Chemical Classes' in item4str:
                                                item4str = item4str.split('->')[1].strip()
                                                result['Safer_Chemical_Classes'] = item4str
                                            else:
                                                usemanu.append(item4str)
                                        usemanu = _cleanandlower_list(usemanu)
                                        result['Use_Classification'] = list(set(usemanu))

                                    if item3.get('TOCHeading') == 'Industry Uses':
                                        usemanu = []
                                        for item4 in item3['Information']:
                                            for item5 in item4['Value']['StringWithMarkup']:
                                                usemanu.append(item5['String'])
                                        usemanu = _cleanandlower_list(usemanu)
                                        result['Industry_Uses'] = list(set(usemanu))
                        
                                    if item3.get('TOCHeading') == 'Consumer Uses':
                                        usemanu = []
                                        for item4 in item3['Information']:
                                            for item5 in item4['Value']['StringWithMarkup']:
                                                usemanu.append(item5['String'])
                                        usemanu = _cleanandlower_list(usemanu)
                                        result['Consumer_Uses'] = list(set(usemanu))
                        
                        if item2.get('TOCHeading') in _temp_tochead:
                            result['Manufacturing'] = 'Available'

                if item.get('TOCHeading') == 'Toxicity':
                    for item2 in item['Section']:
                        pass
                        if item2.get('TOCHeading') == 'Toxicological Information':
                            if item2.get('Section') is not None:
                                for item3 in item2['Section']:
                                    pass
                        if item2.get('TOCHeading') == 'Ecological Information':
                            if item2.get('Section') is not None:
                                for item3 in item2['Section']:
                                    pass
                            
                if item.get('TOCHeading') == 'Classification':
                    for item2 in item['Section']:
                        if item2.get('TOCHeading') == 'NORMAN Suspect List Exchange Classification':
                            norman_list = self._fetch_pubchem_classification_one(cid, hid = 101, cache_folder=folder_path)
                            if (norman_list is not None) and (len(norman_list) > 0):
                                result['NORMAN_Suspect_List_Exchange_Classification'] = norman_list

                result['cid'] = str(int(cid))

            results.append(result)
        results = pd.DataFrame(results)
        # print(results.columns)
        query_df = pd.merge(query_df, results, on='cid', how='left')

        return query_df

#endregion
