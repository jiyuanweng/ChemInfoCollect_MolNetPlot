#region calc mol similarity
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit.Chem import rdFMCS  

def calculate_mcs_similarity(mol1, mol2): 
    mcs_result = rdFMCS.FindMCS([mol1, mol2])  
    
    mcs_size = mcs_result.numAtoms
    total_atoms1 = mol1.GetNumAtoms()  
    total_atoms2 = mol2.GetNumAtoms()  
    
    similarity = (2 * mcs_size) / (total_atoms1 + total_atoms2)
    return similarity  

def tanimoto_coefficient(A, B): 
    # if np.all(A == 0) and np.all(B == 0):  
    #     return 1.0
    intersection = sum(a * b for a, b in zip(A, B))  
    union = sum(A) + sum(B) - intersection  
    return intersection / union if union != 0 else 0.0 

def get_fp_basic(mol):
    if mol is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
    fp2 = Chem.RDKFingerprint(mol)
    fp1 = np.array(fp1)
    fp2 = np.array(fp2)
    fp = np.concatenate((fp1, fp2))  
    return fp

def mol_simi(mol1, mol2):
    try:
        fp1 = get_fp_basic(mol1)
        fp2 = get_fp_basic(mol2)
    except:
        fp1 = None
        fp2 = None
    if fp1 is None or fp2 is None:
        main_fp_simi = 0.0
        core_fp_simi = 0.0
    else:
        main_fp_simi = tanimoto_coefficient(fp1, fp2)
        try:
            core1 = MurckoScaffold.GetScaffoldForMol(mol1)
            core2 = MurckoScaffold.GetScaffoldForMol(mol2)

            if core1.GetNumAtoms() == 0: 
                core1 = mol1
            if core2.GetNumAtoms() == 0:
                core2 = mol2
                
            # core1_smi = MurckoScaffold.MurckoScaffoldSmiles(mol = mol1)
            # core2_smi = MurckoScaffold.MurckoScaffoldSmiles(mol = mol2)
            
            # if core1_smi == '':
            #     core1 = mol1
            # else:
            #     core1 = Chem.MolFromSmiles(core1_smi)
            # if core2_smi == '':
            #     core2 = mol2
            # else:
            #     core2 = Chem.MolFromSmiles(core2_smi)

            fp1 = get_fp_basic(core1)
            fp2 = get_fp_basic(core2)
            core_fp_simi = tanimoto_coefficient(fp1, fp2)
        except:
            core_fp_simi = 0.0
    try:
        msc_simi = calculate_mcs_similarity(mol1, mol2)
    except:
        msc_simi = 0.0

    # similarity = 0.9 * similarity + 0.1 * msc_simi
    return main_fp_simi, core_fp_simi,  msc_simi

def get_fp(mol):
    if mol is None:
        return None
    fp = get_fp_basic(mol)
    core = MurckoScaffold.GetScaffoldForMol(mol)
    fp_core = get_fp_basic(core)
    fp = np.concatenate((fp, fp_core))
    return fp

from joblib import Parallel, delayed
from tqdm import tqdm 
import pickle
import os

def mol_simi_df_ready_task(df, column_name):
    column_values = df[column_name]
    num_rows = len(df)
    tasks = []
    total_tasks_count = num_rows * (num_rows - 1) // 2
    with tqdm(total=total_tasks_count, desc="Generating Tasks") as pbar:
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                index1 = df.index[i]
                index2 = df.index[j]
                value1 = column_values.iloc[i]
                value2 = column_values.iloc[j]
                if value1 is None or value2 is None:
                    continue
                dict0 = {'index1':index1, 'index2':index2,'value1':value1, 'value2':value2,}
                tasks.append(dict0) 
                pbar.update(1)
            
    return tasks

def mol_simi_task(tasks, n_jobs=-1, del_tmp = True, batch_size = 2500000):

    batch_size = batch_size
    if len(tasks) > batch_size:
        
        tmp_dir = 'simi_tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        
        n_batches = len(tasks) // batch_size + (len(tasks) % batch_size > 0)
        for i in range(n_batches):
            # if i < 7: continue # override
            batch_tasks = tasks[i * batch_size:(i + 1) * batch_size]
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(_mol_simi_once)(task_args)
                for task_args in tqdm(batch_tasks, total=len(batch_tasks), desc=f"Processing Batch {i + 1}/{n_batches}",leave=False)
            )
            batch_results = pd.DataFrame(batch_results)
            batch_results.to_feather(f'{tmp_dir}/tmp_{i}.feather')
            # with open(f'{tmp_dir}/tmp_{i}.pkl', 'wb') as f:
            #     pickle.dump(batch_results, f)

        # results = []
        # for i in range(n_batches):
        #     with open(f'{tmp_dir}/tmp_{i}.pkl', 'rb') as f:
        #         batch_results = pickle.load(f)
        #         results.extend(batch_results)
            
        # with open(f'{tmp_dir}/tmp.pkl', 'wb') as f:
        #     pickle.dump(results, f)    
        # results = pd.DataFrame(results)

        results = pd.concat([pd.read_feather(f'{tmp_dir}/tmp_{i}.feather') for i in range(n_batches)])
        results.to_feather(f'{tmp_dir}/tmp.feather')

        if del_tmp:
            for i in range(n_batches):
                # os.remove(f'{tmp_dir}/tmp_{i}.pkl')
                os.remove(f'{tmp_dir}/tmp_{i}.feather')
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_mol_simi_once)(task_args)
            for task_args in tqdm(tasks, total=len(tasks), desc="Processing Pairs")
        )
        results = pd.DataFrame(results)

    return results

def mol_simi_df(df, column_name, n_jobs=-1, del_tmp = True):
    tasks = mol_simi_df_ready_task(df, column_name)
    results = mol_simi_task(tasks, n_jobs, del_tmp)
    return results

def _mol_simi_once(task_args):
    index1 = task_args['index1']
    index2 = task_args['index2']
    value1 = task_args['value1']
    value2 = task_args['value2']
    result_value = mol_simi(value1, value2)
    return {'index1': index1, 'index2': index2, 'simi': result_value}

#endregion

#region drawer
import os
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDraw2DCairo
from io import BytesIO
from PIL import Image, ImageChops, ImageFilter, ImageColor
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import deque
import matplotlib.pyplot as plt

def remove_white_background(img_data, expand_pixels=3, white_threshold=240, preserve_inner_white=True):
    # Convert image to RGBA format
    img = img_data.convert("RGBA")
    width, height = img.size

    # Create a mask for the background
    background_mask = Image.new('L', (width, height), 0)
    pixels = img.load()
    mask_pixels = background_mask.load()

    if preserve_inner_white:
        # Use flood fill to detect white background from edges
        queue = deque()

        # Add edge pixels to the queue if they are white
        for x in range(width):
            if sum(pixels[x, 0][:3]) // 3 > white_threshold:
                queue.append((x, 0))
            if sum(pixels[x, height - 1][:3]) // 3 > white_threshold:
                queue.append((x, height - 1))
        for y in range(height):
            if sum(pixels[0, y][:3]) // 3 > white_threshold:
                queue.append((0, y))
            if sum(pixels[width - 1, y][:3]) // 3 > white_threshold:
                queue.append((width - 1, y))

        # Perform flood fill to mark white background
        while queue:
            x, y = queue.popleft()
            if mask_pixels[x, y] == 255:  # Already processed
                continue
            r, g, b, _ = pixels[x, y]
            if r > white_threshold and g > white_threshold and b > white_threshold:
                mask_pixels[x, y] = 255  # Mark as background
                # Add neighboring pixels
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and mask_pixels[nx, ny] == 0:
                        queue.append((nx, ny))
    else:
        # Detect all white pixels regardless of position
        for x in range(width):
            for y in range(height):
                r, g, b, _ = pixels[x, y]
                if r > white_threshold and g > white_threshold and b > white_threshold:
                    mask_pixels[x, y] = 255  # Mark as background

    # Invert the mask to get the foreground
    foreground_mask = ImageChops.invert(background_mask)

    # Optionally expand the foreground mask
    if expand_pixels > 0:
        kernel_size = 1 + 2 * expand_pixels
        expanded_foreground_mask = foreground_mask.filter(
            ImageFilter.MaxFilter(kernel_size))
    else:
        expanded_foreground_mask = foreground_mask

    # Invert the expanded foreground mask to get the final background mask
    final_background_mask = ImageChops.invert(expanded_foreground_mask)

    # Create a copy of the original image
    output_img = img.copy()
    output_pixels = output_img.load()
    final_mask_pixels = final_background_mask.load()

    # Apply the mask to set background pixels to transparent
    for x in range(width):
        for y in range(height):
            if final_mask_pixels[x, y] > 0:
                output_pixels[x, y] = (0, 0, 0, 0)

    return output_img

def add_stroke(image: Image.Image,
                           stroke_color: str = 'red',
                           stroke_width: int = 3) -> Image.Image:

    color_strs = [c.strip() for c in stroke_color.split(',') if c.strip()]
    rgbs = [ImageColor.getrgb(c) for c in color_strs]
    n_colors = len(rgbs)

    image = image.convert('RGBA')
    ow, oh = image.size
    new_w = ow + 2*stroke_width
    new_h = oh + 2*stroke_width
    canvas = Image.new('RGBA', (new_w, new_h), (0,0,0,0))
    canvas.paste(image, (stroke_width, stroke_width))

    alpha = canvas.split()[3]
    mask_dilated = alpha.filter(ImageFilter.MaxFilter(2*stroke_width+1))
    mask = ImageChops.subtract(mask_dilated, alpha)
    mask = mask.point(lambda x: 255 if x>0 else 0)

    mask_np = np.array(mask)  # shape (H,W), 0 or 255
    yy, xx = np.indices(mask_np.shape)
    cx = new_w/2 - 0.5
    cy = new_h/2 - 0.5
    dx = xx - cx
    dy = cy - yy
    theta = np.degrees(np.arctan2(dy, dx))
    angle = (90.0 - theta) % 360.0

    seg = 360.0 / n_colors
    idx = np.floor(angle / seg).astype(int) % n_colors

    stroke_arr = np.zeros((new_h, new_w, 4), dtype=np.uint8)

    stroke_positions = mask_np == 255
    for i, rgb in enumerate(rgbs):
        sel = (idx == i) & stroke_positions
        stroke_arr[sel, 0] = rgb[0]
        stroke_arr[sel, 1] = rgb[1]
        stroke_arr[sel, 2] = rgb[2]
        stroke_arr[sel, 3] = 255

    stroke_layer = Image.fromarray(stroke_arr, 'RGBA')

    result = Image.alpha_composite(stroke_layer, canvas)
    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)
    return result

def get_mol_image_basic(mol, fix=False, atom_font_size=None, bond_length=None):
    if mol is None:
        return None
    if fix:
        drawer = MolDraw2DCairo(200, 200)
    else:
        drawer = MolDraw2DCairo(-1, -1)
    options = drawer.drawOptions()
    if atom_font_size is not None:
        options.atomLabelFontSize = atom_font_size
    if bond_length is not None:
        options.fixedBondLength = bond_length
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    png = Image.open(BytesIO(png))
    return png

def get_mol_image(mol, fix=False, remove_bg=True, preserve_inner_white=True):
    if mol is None:
        return None
    img = get_mol_image_basic(mol, fix=fix)
    if remove_bg:
        img = remove_white_background(img, preserve_inner_white=preserve_inner_white)
    return img

#endregion