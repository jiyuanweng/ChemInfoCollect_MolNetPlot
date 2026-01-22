#region plot function
import matplotlib.image as mpimg
from matplotlib.patheffects import withStroke  
import shutil
import os
import matplotlib.transforms as mtrans
from matplotlib.patches import Rectangle
import random
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm.auto import tqdm

from deps.help_func_mol import add_stroke

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def interpolate_color(color1, color2, fraction):
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    r = int(r1 + (r2 - r1) * fraction)
    g = int(g1 + (g2 - g1) * fraction)
    b = int(b1 + (b2 - b1) * fraction)
    return rgb_to_hex((r, g, b))

import numpy as np
from matplotlib import colors

import textwrap  
def auto_wrap_text(text, width = 50): 
    wrapped_lines = []  
    for line in text.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=width))  
    return '\n'.join(wrapped_lines)  

from typing import Union, List

def generate_halo(
    halo_color: str,
    halo_length: Union[int, str]
) -> np.ndarray:
    
    colors_list: List[str] = [c.strip() for c in halo_color.split(',') if c.strip()]
    rgbs = np.array([colors.to_rgb(c) for c in colors_list])  # shape (n_colors,3)
    n_colors = len(rgbs)
    if n_colors == 0:
        raise ValueError("error: halo_color")

    if isinstance(halo_length, str) and (',' not in halo_length):
        try:
            halo_length = float(halo_length)
        except ValueError:
            raise ValueError("error: halo_length")

    if isinstance(halo_length, (int, float)):
        radii = np.array([float(halo_length)] * n_colors)
    else:
        radii_str = [s.strip() for s in str(halo_length).split(',') if s.strip()]
        if len(radii_str) != n_colors:
            raise ValueError("error: halo_length")
        radii = np.array([float(v) for v in radii_str])
    max_r = radii.max()
    if max_r <= 0:
        raise ValueError("error: halo_length")

    size = int(2 * max_r)
    center = max_r - 0.5

    y, x = np.ogrid[0:size, 0:size]
    dx = x - center
    dy = center - y
    dist = np.hypot(dx, dy)

    theta = np.degrees(np.arctan2(dy, dx))
    angle = (90.0 - theta) % 360.0

    seg = 360.0 / n_colors
    eps = 1e-6
    idx = np.floor((angle + eps) / seg).astype(int) % n_colors

    this_r = radii[idx]
    alpha = np.clip(1.0 - dist/this_r, 0.0, 1.0)

    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[..., 0] = (rgbs[idx, 0] * 255).astype(np.uint8)
    img[..., 1] = (rgbs[idx, 1] * 255).astype(np.uint8)
    img[..., 2] = (rgbs[idx, 2] * 255).astype(np.uint8)
    img[..., 3] = (alpha * 255).astype(np.uint8)

    return img

def set_equal_aspect_with_padding(ax, pad_pixels=10):
    
    ax.figure.canvas.draw()
    
    data_lim = ax.dataLim
    x_min, x_max = data_lim.intervalx
    y_min, y_max = data_lim.intervaly
    
    ax_bbox = ax.get_window_extent()
    
    data_width = x_max - x_min
    data_height = y_max - y_min
    if data_width <= 0:
        data_width = 1e-9
    if data_height <= 0:
        data_height = 1e-9
    
    pixels_per_unit_x = ax_bbox.width / data_width
    pixels_per_unit_y = ax_bbox.height / data_height
    
    pad_x = pad_pixels / pixels_per_unit_x
    pad_y = pad_pixels / pixels_per_unit_y
    
    new_xlim = (x_min - pad_x, x_max + pad_x)
    new_ylim = (y_min - pad_y, y_max + pad_y)
    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)
    
    ax.set_aspect('equal', adjustable='box')
    
    ax.figure.canvas.draw()
    return ax

def plot_graph(ax, pos: dict, subG, exclude_nodes : list = None, node_image=True,
               node_image_scale=0.5, node_image_max = 150, 
               node_image_stroke = False, 
               node_halo = False, node_halo_radius = 50,
               
               node_label = True, label_fontsize=6, node_color = False,
               
               edge_color = True, edge_label = True, 
               edge_width_scale=5, ax_border = False):

    all_nodes = {str(node) for node in subG.nodes()}

    if len(all_nodes) == 0:
        return

    if exclude_nodes is None:
        exclude_nodes = []

    max_x_pos = max(val[0] for val in pos.values())
    max_y_pos = max(val[1] for val in pos.values())
    min_x_pos = min(val[0] for val in pos.values())
    min_y_pos = min(val[1] for val in pos.values())

    X_pos = {node: pos[node][0] if node in pos else random.uniform(min_x_pos, max_x_pos) for node in all_nodes} 
    Y_pos = {node: pos[node][1] if node in pos else random.uniform(min_y_pos, max_y_pos) for node in all_nodes} 

    # max_x_pos = max(X_pos.values())
    # max_y_pos = max(Y_pos.values())
    # min_x_pos = min(X_pos.values())
    # min_y_pos = min(Y_pos.values())
    # print(min_x_pos, max_x_pos, min_y_pos, max_y_pos)
    
    stroke_effect = withStroke(linewidth=2, foreground="white")  

    for node, attr in tqdm(subG.nodes(data=True), desc='Ploting nodes',leave=False):
        
        x, y = X_pos[node], Y_pos[node]
        color = 'grey'

        if node_color:
            color = attr['color']

        if node in exclude_nodes:
            ax.plot(x, y,         
                marker='x',
                markersize=5,
                markerfacecolor=color,
                markeredgecolor='grey',
                markeredgewidth=1,
                alpha = 1,
                zorder = 2)
            continue

        ax.plot(x, y,         
                marker='x',
                markersize=5,
                markerfacecolor=color,
                markeredgecolor='grey',
                markeredgewidth=1 ,zorder = 2)
        
        if node_halo:
            color = attr.get('color', 'blue')
            halo_radius = attr.get('aura_radius', node_halo_radius)
            halo_img = generate_halo(halo_color = color, halo_length=halo_radius)
            offset_img = OffsetImage(halo_img, zoom=1)
            ab = AnnotationBbox(offset_img, (x,y), frameon=False)
            ab.set_zorder(1)
            ax.add_artist(ab)

        scaled_height = 100
        if node_image:
            img_path = attr.get('image', None)
            if img_path is not None:

                if os.path.exists(img_path):

                    # img = mpimg.imread(img_path)
                    img = Image.open(img_path)

                    if node_image_stroke:
                        color = attr['color']
                        img = add_stroke(img, color, 1)
                    
                    img = np.array(img)

                    img_width, img_height = img.shape[1], img.shape[0]
                    scaled_width = img_width * node_image_scale
                    scaled_height = img_height * node_image_scale
                    
                    if scaled_width > node_image_max or scaled_height > node_image_max:
                        scale_factor = node_image_max / max(scaled_width, scaled_height)
                        scaled_width *= scale_factor
                        scaled_height *= scale_factor
                        
                        scale_factor_total = node_image_scale * scale_factor
                    else:
                        scale_factor_total = node_image_scale

                    imagebox = OffsetImage(img, zoom=scale_factor_total, interpolation="none", alpha=1 )

                    ab = AnnotationBbox(imagebox, (x, y), frameon=False, box_alignment=(0.5, 0.5))
                    ab.set_zorder(3)
                    ax.add_artist(ab)

                    # ax.imshow(img, extent=(x - scaled_width / 2, x + scaled_width / 2, 
                    #                         y - scaled_height / 2, y + scaled_height / 2), zorder=3, interpolation="none")

        if node_label:
            if not node_image:
                scaled_height = 100
            label = attr['label']
            label = auto_wrap_text(label)
            trans_offset = mtrans.offset_copy(ax.transData, fig=ax.figure, x=0, y= - scaled_height / 2, units='points')
            ax.text(x, y, label, ha='center', va='top', fontsize=label_fontsize, zorder=4, transform=trans_offset, path_effects=[stroke_effect])
            
            # ax.text(x, y - scaled_height / 2 - 50, label, ha='center', va='bottom', fontsize=label_fontsize, zorder=4, path_effects=[stroke_effect]) 

    for u, v, attr in tqdm(subG.edges(data=True),desc='Ploting edges',leave=False):
        x1, y1 = X_pos[u], Y_pos[u]
        x2, y2 = X_pos[v], Y_pos[v]

        if (u in exclude_nodes) or (v in exclude_nodes):
            continue

        simi = attr['simi']
            
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2


        if edge_color:
            edge_color = attr.get('color', '#232323')
            edge_width = simi * edge_width_scale
        
        else:
            edge_color = "#232323"
            edge_width = 1 * edge_width_scale

        ax.plot([x1, x2], [y1, y2], 
                        color=edge_color, 
                        linewidth=edge_width, 
                        solid_capstyle='round', zorder=1)
        
        if edge_label:
            ax.text(mid_x, mid_y, f'{simi:.2f}', ha='center', va='center', fontsize=10, color='black', zorder=4, path_effects=[stroke_effect]) 

    # padding = 100
    # x_limits = ax.get_xlim()
    # y_limits = ax.get_ylim()
    # ax.set_xlim(x_limits[0] - padding, x_limits[1] + padding)
    # ax.set_ylim(y_limits[0] - padding, y_limits[1] + padding)

    # ax.axis('tight')
    set_equal_aspect_with_padding(ax, pad_pixels=node_image_max)
    ax.axis('off')

    if ax_border:
        border_color = 'black'  
        border_width = 2  

        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.spines['top'].set_color(border_color)
        ax.spines['bottom'].set_color(border_color)
        ax.spines['left'].set_color(border_color)
        ax.spines['right'].set_color(border_color)
        ax.spines['top'].set_linewidth(border_width)
        ax.spines['bottom'].set_linewidth(border_width)
        ax.spines['left'].set_linewidth(border_width)
        ax.spines['right'].set_linewidth(border_width)

        rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=border_color, linewidth=border_width)
        ax.add_patch(rect)

        # fig.canvas.draw()
        # ax_pos = ax.get_position()
        # margin = 0.00
        # rect_x0 = ax_pos.x0 - margin
        # rect_y0 = ax_pos.y0 - margin
        # rect_width = ax_pos.width + 2 * margin
        # rect_height = ax_pos.height + 2 * margin
        # rect = Rectangle((rect_x0, rect_y0), rect_width, rect_height, transform=fig.transFigure, fill=False, edgecolor=border_color, linewidth=border_width, zorder=10)
        # fig.add_artist(rect)

    ax.set_facecolor('white')

#endregion

#region trim edge on top simi for each node
from tqdm.auto import tqdm

def preprocess_results_top_n_edges_enhan(results, simi_threshold=0.5,top_n=1,backward=False,global_top=None,global_threshold=None,mute=False):
    # filter and sort edges by similarity descending, then by indices
    filtered = [e for e in results if e['simi'] > simi_threshold]
    filtered.sort(key=lambda x: (-x['simi'], x['index1'], x['index2']))

    # collect edges per node
    node_edges = {}
    for e in filtered:
        node_edges.setdefault(e['index1'], []).append(e)
        node_edges.setdefault(e['index2'], []).append(e)

    # pre-sort each node's edges for deterministic top-n selection
    for edges in node_edges.values():
        edges.sort(key=lambda x: (-x['simi'], x['index1'], x['index2']))

    def scan(nodes):
        seen = set()
        selected = []
        for node in nodes:
            count = 0
            for e in node_edges.get(node, []):
                key = tuple(sorted((e['index1'], e['index2'])))
                if key not in seen:
                    seen.add(key)
                    selected.append(e)
                    count += 1
                    if count >= top_n:
                        break
        return selected, seen

    # forward scan
    nodes = sorted(node_edges)
    forward_sel, seen = scan(nodes)

    # optional backward scan
    results_sel = forward_sel.copy()
    if backward:
        backward_sel, back_seen = scan(reversed(nodes))
        # merge without duplicates
        for e in backward_sel:
            key = tuple(sorted((e['index1'], e['index2'])))
            if key not in seen:
                seen.add(key)
                results_sel.append(e)

    # global additions
    if global_top is not None:
        for e in filtered[:global_top]:
            key = tuple(sorted((e['index1'], e['index2'])))
            if key not in seen:
                seen.add(key)
                results_sel.append(e)

    if global_threshold is not None:
        for e in filtered:
            if e['simi'] >= global_threshold:
                key = tuple(sorted((e['index1'], e['index2'])))
                if key not in seen:
                    seen.add(key)
                    results_sel.append(e)

    if not mute:
        print(f"Raw: {len(filtered)}, Filtered: {len(results_sel)}")
    return results_sel

#endregion

#region calc pos with pyvis - rounder
from pyvis.network import Network
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import json
import os
from tqdm.auto import tqdm
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

# Wait until the injected script sets window.networkStable
def wait_for_network_stability(driver, timeout=30, poll_frequency=3, strict_timeout=True, silent=True):
    try:
        WebDriverWait(driver, timeout, poll_frequency=poll_frequency).until(
            lambda d: d.execute_script("return window.networkStable === true;")
        )
        if not silent:
            tqdm.write("Network stabilized")
        return 0
    except TimeoutException:
        if strict_timeout:
            raise TimeoutException("Network did not stabilize in time")
        else:
            if not silent:
                tqdm.write("Timeout reached, proceeding anyway")
            return 1
        
# Inject JS listener into HTML content based on event type
def inject_event_listener(html_content, event_type="stabilizationIterationsDone"):
    if event_type == "stabilized":
        js = """
            <script>
            // Debug: log when waiting for full stabilization
            console.log("Waiting for 'stabilized' event");
            network.once('stabilized', function() {
                console.log("'stabilized' event fired");
                window.networkStable = true;
            });
            </script>
            """
    elif event_type == "stabilizationIterationsDone":
        js = """
            <script>
            // Debug: log when first stabilization iteration ends
            console.log("Waiting for first 'stabilizationIterationsDone' event");
            network.once('stabilizationIterationsDone', function(iterations) {
                console.log("'stabilizationIterationsDone' event fired, iterations left: " + iterations);
                window.networkStable = true;
            });
            </script>
            """
    else:
        raise ValueError("Invalid event_type specified")
    return html_content.replace("</body>", js + "\n</body>")


def pyvis_position_round(G, init_pos : dict = None, html_name : str = None, fast : bool = False, keep_html : bool = False, timeout : int = 30, init_timeout = 120, script_timeout = 30):
    
    waiting_times = [3,2]
    if fast:
        waiting_times = [1,1]
    
    if html_name is None:
        html_name = 'temp.html'

    steps = ['Writing pyvis html...', 
             'Opening html file...', 
             'Waiting for stability...', 
             'First time modifying physics setting...',
             'Second time modifying physics setting...',
             'Third time modifying physics setting...',
             'Saving positions...']

    with tqdm(total = len(steps), leave = False) as pbar:

        pbar.set_description(steps[0])
        net = Network()
        net.from_nx(G)
        net.toggle_physics(True)
        net.show_buttons(filter_=[
        # 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 
        'physics', 
        # 'selection', 'renderer'
        ])

        # net.options = {
        #     "layout": {
        #         "randomSeed": 12345
        #     }
        # }
        
        if init_pos is not None:
            for node in net.nodes: 
                node['x'], node['y'] = init_pos[node['id']][0], init_pos[node['id']][1]

        net.force_atlas_2based(gravity=-200, central_gravity=0.01, spring_length=0, spring_strength=1, damping=0.2, overlap=0.8)
        
        # net.set_options = {
        #     "physics": {
        #         "forceAtlas2Based": {
        #             "gravitationalConstant": -200,
        #             "centralGravity": 0.01,
        #             "springLength": 0,
        #             "springConstant": 1,
        #             "damping": 0.2,
        #             "avoidOverlap": 0.8
        #         },
        #         "solver": "forceAtlas2Based"
        #     },
        #     "layout": {
        #         "randomSeed": 42
        #     }
        # }
        
        net.write_html(html_name)

        with open(html_name, "r", encoding="utf-8") as f:
            content = f.read()
        content = inject_event_listener(content)
        with open(html_name, "w", encoding="utf-8") as f:
            f.write(content)

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(init_timeout)
        driver.set_script_timeout(script_timeout)
        
        current_dir = os.getcwd()
        html_path = os.path.join(current_dir, html_name)
        pbar.update(1)
        pbar.set_description(steps[1])
        driver.get(html_path)
        
        pbar.update(1)
        pbar.set_description(steps[2])

        init_stat_code = wait_for_network_stability(driver, timeout=timeout, poll_frequency=waiting_times[0], strict_timeout=False)
        time.sleep(waiting_times[0])

        pbar.update(1)
        pbar.set_description(steps[3])
        driver.execute_script('''
            return network.setOptions({
            physics: {
                forceAtlas2Based: {
                springLength: 50,
                }
            }
            });
            ''')
        time.sleep(waiting_times[1])

        pbar.update(1)
        pbar.set_description(steps[4])
        driver.execute_script('''
            return network.setOptions({
            physics: {
                forceAtlas2Based: {
                springLength: 100,
                }
            }
            });
            ''')
        time.sleep(waiting_times[1])

        pbar.update(1)
        pbar.set_description(steps[5])
        driver.execute_script('''
            return network.setOptions({
            physics: {
                forceAtlas2Based: {
                springLength: 200,
                springConstant: 0.5,
                }
            }
            });
            ''')
        time.sleep(waiting_times[1])

        pbar.update(1)
        pbar.set_description(steps[6])
        positions = driver.execute_script("return JSON.stringify(network.getPositions());")
        positions = json.loads(positions)
        pos = {str(node): (data['x'], data['y']) for node, data in positions.items()}
        
        driver.quit()

        if not keep_html:
            os.remove(html_path)
        else:
            for node in net.nodes:
                node_id = str(node['id'])
                if node_id in positions:
                    pos_pyvis = positions[node_id]
                    node['x'] = pos_pyvis['x']
                    node['y'] = pos_pyvis['y']
                    # node['fixed'] = True
            # net.force_atlas_2based(spring_length=200, spring_strength=0.5)
            net.set_options = {
                "physics": {
                    "forceAtlas2Based": {
                    "gravitationalConstant": -200,
                    "springLength": 150,
                    "springConstant": 0.6,
                    "damping": 0.2,
                    "avoidOverlap": 0.8
                    },
                    "minVelocity": 0.75,
                    "solver": "forceAtlas2Based"
                }
                }
            net.toggle_physics(False)
            net.write_html(html_name)

        return pos
    
#endregion

#region calc pos with pyvis
from pyvis.network import Network
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import json
import os
from tqdm.auto import tqdm
from selenium.webdriver.support.ui import WebDriverWait

def pyvis_position(G, init_pos : dict = None, html_name : str = None, fast : bool = False, keep_html : bool = False, timeout : int = 30, init_timeout = 120,script_timeout = 30):
    
    waiting_times = [7,3]
    if fast:
        waiting_times = [1,1]
    
    if html_name is None:
        html_name = 'temp.html'

    steps = ['Writing pyvis html...', 
             'Opening html file...', 
             'Waiting for stability...', 
             'First time modifying physics setting...',
             'Second time modifying physics setting...',
             'Third time modifying physics setting...',
             'Saving positions...']

    with tqdm(total = len(steps), leave = False) as pbar:

        pbar.set_description(steps[0])
        net = Network()
        net.from_nx(G)
        net.toggle_physics(True)
        net.show_buttons(filter_=[
        # 'nodes', 'edges', 'layout', 'interaction', 'manipulation', 
        'physics', 
        # 'selection', 'renderer'
        ])

        # net.set_options = {
        #             "layout": {
        #                 "randomSeed": 42
        #             }
        #         }
        
        if init_pos is not None:
            for node in net.nodes: 
                node['x'], node['y'] = init_pos[node['id']][0], init_pos[node['id']][1]

        net.repulsion(node_distance=300, central_gravity=0.05, spring_length=0, spring_strength=1, damping=0.5)
        
        # net.set_options = {
        #     "layout": {
        #         "randomSeed": 42
        #     },
        #     "physics": {
        #         "repulsion": {
        #             "nodeDistance": 300,
        #             "centralGravity": 0.05,
        #             "springLength": 0,
        #             "springConstant": 1,
        #             "damping": 0.5
        #         }
        #     }
        # }
        
        net.write_html(html_name)

        with open(html_name, "r", encoding="utf-8") as f:
            content = f.read()
        content = inject_event_listener(content)
        with open(html_name, "w", encoding="utf-8") as f:
            f.write(content)

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(init_timeout)
        driver.set_script_timeout(script_timeout)
        current_dir = os.getcwd()
        html_path = os.path.join(current_dir, html_name)
        pbar.update(1)
        pbar.set_description(steps[1])
        driver.get(html_path)
        
        pbar.update(1)
        pbar.set_description(steps[2])
        
        init_stat_code = wait_for_network_stability(driver, timeout=timeout, poll_frequency=waiting_times[0], strict_timeout=False)
        time.sleep(waiting_times[0])
        
        pbar.update(1)
        pbar.set_description(steps[3])
        driver.execute_script('''
            return network.setOptions({
            physics: {
                repulsion: {
                springLength: 100,
                }
            }
            });
            ''')
        time.sleep(waiting_times[1])

        pbar.update(1)
        pbar.set_description(steps[4])
        driver.execute_script('''
            return network.setOptions({
            physics: {
                repulsion: {
                springLength: 200,
                }
            }
            });
            ''')
        time.sleep(waiting_times[1])

        pbar.update(1)
        pbar.set_description(steps[5])
        driver.execute_script('''
            return network.setOptions({
                physics: {
                    enabled: true,
                    solver: 'barnesHut',
                    barnesHut: {
                    theta: 0.5,
                    gravitationalConstant: -2000,
                    centralGravity: 0.1,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.4,
                    avoidOverlap: 1
                    },
                }
                });
            ''')
        time.sleep(waiting_times[1])

        pbar.update(1)
        pbar.set_description(steps[6])
        positions = driver.execute_script("return JSON.stringify(network.getPositions());")
        positions = json.loads(positions)
        pos = {str(node): (data['x'], data['y']) for node, data in positions.items()}
        
        driver.quit()

        if not keep_html:
            os.remove(html_path)
        else:
            for node in net.nodes:
                node_id = str(node['id'])
                if node_id in positions:
                    pos_pyvis = positions[node_id]
                    node['x'] = pos_pyvis['x']
                    node['y'] = pos_pyvis['y']
                    # node['fixed'] = True

            net.barnes_hut(gravity=-2000, spring_length=150, spring_strength=0.04, damping=0.4, overlap=1)
            # net.set_options = {
            #     "physics": {
            #         "barnesHut": {
            #         "gravitationalConstant": -2000,
            #         "centralGravity": 0.4,
            #         "springLength": 150,
            #         "springConstant": 0.05
            #         },
            #         "minVelocity": 0.75
            #     }
            #     }
            net.toggle_physics(False)
            net.write_html(html_name)

        return pos
#endreigon

#region trim edge and layout
import math
import networkx as nx
import matplotlib.pyplot as plt
import zlib

def deterministic_subgraph(G, nodes = None):
    if nodes is None:
        nodes = list(G.nodes)
    sorted_nodes = sorted(nodes)
    SG = nx.Graph()
    for node in sorted_nodes:
        SG.add_node(node, **G.nodes[node])
    node_set = set(sorted_nodes)
    edges = []
    for u, v in G.edges():
        if u in node_set and v in node_set:
            edge = tuple(sorted((u, v)))
            edges.append((edge[0], edge[1], G.edges[u, v]))
    edges.sort()
    for u, v, attr in edges:
        SG.add_edge(u, v, **attr)
    return SG

def deterministic_offset(u, v):
    sorted_nodes = sorted([str(u), str(v)])
    unique_str = "_".join(sorted_nodes)

    crc = zlib.crc32(unique_str.encode()) & 0xffffffff
    return crc / 1e15

def trim_edge(G, keep_top_n = True, keep_span_tree = True, simi_threshold=0.7):   
    
    if len(G.nodes()) == 0 or len(G.nodes()) == 1:
        return G

    G_tempsimi = deterministic_subgraph(G)

    # G_tempsimi = G.copy()
    # G_copy = G.copy()

    edges_list = [
    {'index1': u, 'index2': v, 'simi': data['simi']}
    for u, v, data in G.edges(data=True)
    ]
    edges_list_0 = preprocess_results_top_n_edges_enhan(edges_list, top_n=1, backward=False, mute= True)

    # deal with the same simi
    # for u, v, data in G_tempsimi.edges(data=True):
    #     data['temp_simi'] = data['simi'] + deterministic_offset(u, v)

    prunedG = nx.maximum_spanning_tree(G_tempsimi, weight='simi')

    # edges_list_1 = [
    # {'index1': min(u,v), 'index2': max(u,v), 'simi': data['simi']}
    # for u, v, data in prunedG.edges(data=True)
    # ]
    # edges_list_1.sort(key=lambda x: (x['simi'],x['index1'], x['index2']))

    prunedG = deterministic_subgraph(prunedG)

    edges_list_1 = [
    {'index1': u, 'index2': v, 'simi': data['simi']}
    for u, v, data in prunedG.edges(data=True)
    ]
    
    G_tempsimi.clear_edges()

    if keep_top_n:
        for item in tqdm(edges_list_0, desc='Adding edges', leave=False):
            i = item['index1']
            j = item['index2']
            simi = item['simi']
            if simi <= simi_threshold:
                continue
            
            edge_data = G.get_edge_data(i, j)
            G_tempsimi.add_edge(i, j, **edge_data)

    if keep_span_tree:
        for item in tqdm(edges_list_1, desc='Adding edges', leave=False):
            i = item['index1']
            j = item['index2']
            simi = item['simi']
            if simi <= simi_threshold:
                continue

            edge_data = G.get_edge_data(i, j)
            G_tempsimi.add_edge(i, j, **edge_data)

    G_tempsimi = deterministic_subgraph(G_tempsimi)

    return G_tempsimi

def cal_layout(G, pos= None, scale = 100, keep_html = False, html_name = 'temp.html', timeout = 30, init_timeout = 120, script_timeout = 30):
    count = len(G.nodes())
    scale = count * scale

    if count == 0:
        pos = {}

    elif count == 1:
        index = list(G.nodes())[0]
        pos = {str(index): (0, 0)}

    elif 1 < count <= 2:
        index1 = list(G.nodes())[0]
        index2 = list(G.nodes())[1]
        pos = {str(index1): (0, 0), str(index2): (0, scale)}

    elif 2 < count <= 10:
        # pos = nx.kamada_kawai_layout(G, pos=pos)
        pos = nx.forceatlas2_layout(G, pos=pos, seed=42)
        # pos = nx.spring_layout(G, pos=pos, seed=42)
        for node, (x, y) in pos.items():
            pos[node] = (x * scale, y * scale)
        # pos = nx.rescale_layout_dict(pos, scale=scale)

    elif 10 < count <= 20:
        if pos is None:
            pos = nx.kamada_kawai_layout(G, scale=scale)
        pos = pyvis_position_round(G, init_pos=pos, fast=True, html_name=html_name, keep_html=keep_html, timeout = timeout, init_timeout = init_timeout, script_timeout =script_timeout)

    elif 20 < count <= 50:
        script_timeout = 60
        init_timeout = 120
        timeout = 30
        if pos is None:
            pos = nx.kamada_kawai_layout(G, scale=scale)
        pos = pyvis_position(G, init_pos=pos, fast=True, html_name=html_name, keep_html=keep_html, timeout = timeout, init_timeout = init_timeout, script_timeout =script_timeout)

    else:
        script_timeout = 120
        init_timeout = 240
        timeout = 60
        if pos is None:
            pos = nx.kamada_kawai_layout(G, scale=scale)
        pos = pyvis_position(G, init_pos=pos, html_name=html_name, keep_html=keep_html, timeout = timeout, init_timeout = init_timeout, script_timeout =script_timeout)
         
    return pos

#endregion

#region special spare layout
def compute_grid_layout(clusters, dist=1, sort = True):
    # clusters: list of clusters (each is an iterable of nodes)
    if sort:
        clusters = sorted(clusters, key=lambda c: len(c), reverse=True)
    total_nodes = sum(len(cluster) for cluster in clusters)
    if total_nodes == 0:
        return {}
    # Compute grid width; ensure each cluster can fit in one row
    grid_cols_init = math.ceil(math.sqrt(total_nodes))
    max_cluster_size = max(len(cluster) for cluster in clusters)
    grid_cols = max(grid_cols_init, max_cluster_size)
    
    pos = {}
    cur_row = 0
    cur_col = 0
    for cluster in clusters:
        cluster_size = len(cluster)
        # If not enough columns remain for the whole cluster, jump to next row
        if cur_col + cluster_size > grid_cols:
            cur_row += 1
            cur_col = 0
        # Assign positions for nodes in the cluster (order sorted for consistency)
        for node in sorted(cluster, key=lambda x: str(x)):
            pos[node] = (cur_col * dist, -cur_row * dist)
            cur_col += 1
    return pos

#endregion

#region list utility

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def count_bottom_elements(nested_list):
    total = 0
    for element in nested_list:
        if isinstance(element, (list, set)):
            total += count_bottom_elements(element)
        else:
            total += 1
    return total

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    else:
        return obj

def construct_sublist(full_list, sub_list):
    '''
    a=[[0,1],[[2,3,4],[5]],6]
    b=[0,1,3,6]

    construct_sublist(a,b)
    '''
    sub_set = set(sub_list)
    
    def process(element):
        if isinstance(element, list):
            processed = []
            for e in element:
                result = process(e)
                if result is not None:
                    processed.append(result)
            return processed if processed else []
        else:
            return element if element in sub_set else None
    
    return process(full_list)

#endregion

#region split community
from networkx.algorithms import community

def split_communities_girvan_newman(graph, max_size=250, min_size_threshold=10):
    """
    split community using Girvan-Newman, slow

    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX Graph object.")
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_nodes() <= max_size and nx.is_connected(graph):
         pass

    comp_generator = community.girvan_newman(graph)

    best_partition = None
    initial_components = tuple(nx.connected_components(graph))
    
    if all(len(c) <= max_size for c in initial_components):
        best_partition = list(initial_components) # 使用 frozenset 转换为 set
    else:
        for partition_tuple in comp_generator:
            current_max_size = 0
            if partition_tuple:
                current_max_size = max(len(c) for c in partition_tuple)
            
            if current_max_size <= max_size:
                best_partition = [set(c) for c in partition_tuple] 
                break
            
        if best_partition is None:
            print("Warning: Graph might have been fully decomposed. Returning single-node communities.")
            final_level_partitions = list(community.girvan_newman(graph))
            if final_level_partitions:
                 best_partition = [set(c) for c in final_level_partitions[-1]]
            else:
                 best_partition = [set(c) for c in initial_components]

    small_communities = [c for c in best_partition if len(c) <= min_size_threshold]
    if small_communities:
        print(f"Info: Found {len(small_communities)} communities with size <= {min_size_threshold}.")
        # print(f"Sizes of small communities: {[len(c) for c in small_communities]}")

    return best_partition

def split_by_community(G, sort=True, resolution = 0.2, seed = 42):
    rnd = random.Random(seed)

    communities = list(community.louvain_communities(G, weight='simi', seed=rnd, resolution = resolution))
    # communities = list(community.greedy_modularity_communities(H, weight='temp_simi'))
    if sort:
        communities.sort(key=lambda c: len(c), reverse=True)
    return communities

#endregion

#region plot by group cluster
import io
from PIL import Image
import rectpack
import gc
Image.MAX_IMAGE_PIXELS = 9000000000

def calc_fig_sizes(node_groups, pix_multi = 1000, same_subplot_size=False):
    max_node_count = max(len(cluster) for cluster in node_groups)
    f_sizes = []
    for i, cluster in enumerate(node_groups):
        if same_subplot_size:
            size = math.ceil(math.sqrt(max_node_count))
        else:
            size = math.ceil(math.sqrt(len(cluster)))
        f_sizes.append([size, size])

    f_sizes = [[int(pix_multi*w),int(pix_multi*h)] for w,h in f_sizes]

    return f_sizes

def arrange_subplots_rectpack(imgs, max_width, max_height = float('inf'), sort=True):
    
    if sort:
        imgs = sorted(imgs, key=lambda img: img.size[0] * img.size[1], reverse=True)
    
    packer = rectpack.newPacker(
        mode=rectpack.PackingMode.Offline,
        bin_algo=rectpack.PackingBin.Global,
        pack_algo=rectpack.MaxRectsBl,
        sort_algo=rectpack.SORT_NONE,
        rotation=False
    )
    
    for i, img in enumerate(imgs):
        packer.add_rect(*img.size, rid=i)
    
    packer.add_bin(max_width, max_height)
    
    packer.pack()

    actual_height = 0
    actual_width = 0
    for rect in packer.rect_list():
        b, x, y, w, h, rid = rect
        if y + h > actual_height:
            actual_height = y + h
        if x + w > actual_width:
            actual_width = x + w

    atlas = Image.new('RGBA', ((int(actual_width)), (int(actual_height))))
    for rect in packer.rect_list():
        b, x, y, w, h, rid = rect
        atlas.paste(imgs[rid], (x, y))

    return atlas

def is_nested(obj):
    if isinstance(obj, (list, set)):
        return any(isinstance(item, (list, set)) for item in obj)
    return False

def replace_dicts_with_keys(obj):
    if isinstance(obj, list):
        return [replace_dicts_with_keys(item) for item in obj]
    elif isinstance(obj, dict):
        return list(obj.keys())
    else:
        return obj

def plot_clusters(
    G,
    node_groups,
    folder_path,
    file_num,

    split_plot = False,

    pos_json=None,
    pos_json_control = False,
    save_pos=True,

    same_subplot_size = False,
    simi_threshold = 0.7,
    
    dpi=200,

    pix_multi=1000,

    keep_html=False,
    html_name_tpl=None,
    
    node_image = True,
    node_halo = True,
    node_halo_radius = 100,
    node_label = True,
    edge_color = True,
    edge_label = True,
    image_stroke = True,
    ax_border = False,
    draw_outer_border = False,
):
    
    if pos_json is None or pos_json == [] or pos_json == {}:
        pos_json = None
        pos_json_control = False
        
    if is_nested(node_groups):
        if pos_json:
            if isinstance(pos_json, list) and isinstance(pos_json[0], dict) and len(pos_json) == len(node_groups):
                pass
            else:
                raise ValueError("wrong pos_json format")
            
    else:
        node_groups = [node_groups]
        if pos_json:
            if isinstance(pos_json, dict):
                pos_json = [pos_json]
            else:
                raise ValueError("wrong pos_json format, must be a dict")
            
    if pos_json_control:
        full_node_groups = replace_dicts_with_keys(pos_json)
        f_sizes = calc_fig_sizes(full_node_groups, pix_multi = pix_multi, same_subplot_size=same_subplot_size)
    else:
        f_sizes = calc_fig_sizes(node_groups, pix_multi = pix_multi, same_subplot_size=same_subplot_size)

    pic_cache = []

    pos_json_list = []

    for i, cluster in enumerate(tqdm(node_groups, leave = False)):
        
        if pos_json_control:
            full_cluster = list(pos_json[i].keys())
            node_count = len(full_cluster)
            exclude_nodes = [node for node in full_cluster if node not in cluster]
        else:
            node_count = len(cluster)
            full_cluster = cluster
            exclude_nodes = None

        # print(cluster)
        subG = deterministic_subgraph(G, nodes=full_cluster)
        prunedG = trim_edge(subG, simi_threshold=simi_threshold)

        if not pos_json or not pos_json[i] or pos_json[i] in ({}, []):
            timeout = 60 if node_count <= 100 else (120 if 100 < node_count < 300 else 240)
            if not html_name_tpl:
                html_name_tpl = f'temp_{str(file_num)}_{str(i)}.html'
            pos = cal_layout(prunedG, scale=pix_multi, 
                             keep_html=keep_html, html_name=html_name_tpl, 
                             timeout=timeout, init_timeout=960, script_timeout=120)
        else:
            pos = pos_json[i]
            
        pos_json_list.append(pos)

        f_size = f_sizes[i]
        f_size_w = f_size[0]
        f_size_h = f_size[1]

        fig, ax = plt.subplots(figsize=(f_size_w / dpi, f_size_h / dpi), dpi=dpi)
        plot_graph(ax, pos, prunedG, exclude_nodes=exclude_nodes,
                   node_image=node_image, node_label=node_label,
                   node_image_stroke=image_stroke, 
                   node_halo=node_halo, node_halo_radius=node_halo_radius,
                   edge_color=edge_color, edge_label=edge_label, 
                   ax_border=ax_border)
        
        fig.tight_layout()

        if draw_outer_border:
            global_rect = plt.Rectangle((0, 0), 1, 1,fill=False, edgecolor='black', linewidth=2,transform=fig.transFigure, zorder=5)
            fig.add_artist(global_rect)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, 
        # bbox_inches='tight', pad_inches=0
        )
        buf.seek(0)
        image_data = buf.read()
        img = Image.open(io.BytesIO(image_data))
        pic_cache.append(img)
        plt.close(fig)
        buf.close()
        del buf, fig, ax
        gc.collect()
        
    fig_save_path = f'{folder_path}/fig_{file_num:02d}.png'

    if save_pos:
        if len(pos_json_list) == 1:
            pos_file = pos_json_list[0]
        else:
            pos_file = pos_json_list
        json.dump(pos_file, open(f'{folder_path}/pos_{file_num:02d}.json', 'w'))

    if len(pic_cache) == 1:
        pic_cache[0].save(fig_save_path)
    else:
        if split_plot:
            for i, img in enumerate(pic_cache):
                img.save(f'{folder_path}/fig_{file_num:02d}_{i}.png')
        else:
            # for img in pic_cache:
            #     print(img.size)
            total_area = sum(img.size[0] * img.size[1] for img in pic_cache)
            max_width = int(math.sqrt(total_area))
            if same_subplot_size:
                sort = False
            else:
                sort = True
            atlas = arrange_subplots_rectpack(pic_cache, max_width, sort=sort)
            atlas.save(fig_save_path)
    
    del pic_cache
    gc.collect()

#endregion