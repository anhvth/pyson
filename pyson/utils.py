import json
import numpy as np
from glob import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
from functools import partial
#import mmcv
import numpy as np
from six.moves import map, zip
import inspect
import shutil
try:
    import xxhash
    import torch
except:
    pass

print_source = lambda x: print(inspect.getsource(x))

def lib_reload(some_module):
    import importlib
    return importlib.reload(some_module)

def do_by_chance(chance):
    assert chance > 1 and chance < 100
    return np.random.uniform() < chance/100

def get_paths(directory, input_type='png'):
    """
        Get a list of input_type paths
        params args:
        return: a list of paths
    """
    paths = glob(os.path.join(directory, '*.{}'.format(input_type)))
    assert len(paths) > 0, '\n\tDirectory:\t{}\n\tInput type:\t{} \n num of paths must be > 0'.format(
        dir, input_type)
    print('Found {} files {}'.format(len(paths), input_type))
    return paths

def read_json(path):
    '''Read a json path.
        Arguments: 
            path: string path to json 
        Returns:
             A dictionary of the json file
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def swap_list(lst):
    assert isinstance(lst, (list, tuple))
    assert isinstance(lst[0], (list, tuple))
    len_list = len(lst)
    len_elem = len(lst[0])
    out = [[None for _ in range(len_list)] for _ in range(len_elem)]
    for i in range(len_list):
        for j in range(len_elem):
            out[j][i] = lst[i][j]
    return out


def multi_thread(fn, array_inputs, max_workers=None, desc="Executing Pipeline", unit=" Samples", verbose=False):
    def _wraper(x):
        i, input = x
        return {i: fn(input)}
    
    array_inputs = [(i, _) for i, _ in enumerate(array_inputs)]
    if verbose:
        with tqdm(total=len(array_inputs), desc=desc, unit=unit) as progress_bar:
            outputs = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(_wraper, array_inputs):
                    outputs.update(result)
                    progress_bar.update(1)
    else:
        outputs = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(_wraper, array_inputs):
                outputs.update(result)
    if verbose:
        print('Finished')
    outputs = list(outputs.values())
    return outputs
    
 
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def noralize_filenames(directory, ext='*'):
    paths = glob('{}.{}'.format(directory, ext))
    for i, path in enumerate(paths):
        base_dir, base_name = os.path.split(path)
        name, base_ext = base_name.split('.')
        new_name = '{:0>4d}.{}'.format(i, base_ext)
        new_path = os.path.join(base_dir, new_name)
        print('Rename: {} --> {}'.format(path, new_path))
        os.rename(path, new_path)

def identify(x):
    '''Return an hex digest of the input'''
    return xxhash.xxh64(pickle.dumps(x), seed=0).hexdigest()


        
        
def memoize(func):
    import xxhash
    import pickle
    import os
    from functools import wraps
    '''Cache result of function call on disk
    Support multiple positional and keyword arguments'''

    def print_status(status, func, args, kwargs):
        pass
    

    @wraps(func)
    def memoized_func(*args, **kwargs):
        cache_dir = 'cache'
        try:
            if 'hash_key' in kwargs.keys():
                import inspect                
                func_id = identify(kwargs['hash_key'])
            else:
                import inspect
                func_id = identify((inspect.getsource(func), args, kwargs))
            cache_path = os.path.join(cache_dir, func_id)
            
            if (os.path.exists(cache_path) and
                    not func.__name__ in os.environ and
                    not 'BUST_CACHE' in os.environ):
                return pickle.load(open(cache_path, 'rb'))
            else:
                result = func(*args, **kwargs)
                os.makedirs(cache_dir, exist_ok=True)
                pickle.dump(result, open(cache_path, 'wb'))
                return result
        except (KeyError, AttributeError, TypeError):
            return func(*args, **kwargs)
    return memoized_func


def make_mini_dataset(json_path, image_prefix, out_dir, n=1000):
    print("Making mini dataset", out_dir, "num images:", n)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)
    j = read_json(json_path)
    img_ids = list(set([_["image_id"] for _ in j["annotations"]]))
    img_id2path = {_["id"]:_ for _ in j["images"]}
    images = []
    annotations = []
    print("make images")

    for image in tqdm(j["images"]):
        if image["id"] in img_ids[:n]:
            images.append(image)
            file_name = image["file_name"]
            old_path = os.path.join(image_prefix, file_name)
            new_path = os.path.join(out_dir, "images", file_name)
            shutil.copy(old_path, new_path)

    print("make annotations")
    for annotation in tqdm(j["annotations"]):
        if annotation["image_id"] in img_ids[:n]:
            annotations.append(annotation)
    j["images"] = images
    j["annotations"] = annotations
    out_json = os.path.join(out_dir, "annotations", "mini_json.json")
    with open(out_json, "w") as f:
        json.dump(j, f)
    print(out_json)



def show_df(df, path_column=None, max_col_width=-1):
    """
        Turn a DataFrame which has the image_path column into a html table
            with watchable images
        Argument:
            df: the origin dataframe
            path_column: the column name which contains the paths to images
        Return:
            HTML object, a table with images
    """
    assert path_column is not None, 'if you want to show the image then tell me which column contain the path? if not what the point to use this?'
    import pandas
    from PIL import Image
    from IPython.display import HTML
    from io import BytesIO
    import cv2
    import base64
    
    pandas.set_option('display.max_colwidth', max_col_width)

    def get_thumbnail(path):
        img = cv2.imread(path, 0)
        h,w = img.shape[:2]
        f = 48/h
        img = cv2.resize(img, (0,0), fx=f, fy=f)
        return Image.fromarray(img)

    def image_base64(im):
        if isinstance(im, str):
            im = get_thumbnail(im)
        with BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()

    def image_formatter(im):
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
    
    return HTML(df.to_html(formatters={path_column: image_formatter}, escape=False))




def load_ssh(link, ssh_username, ssh_password, server_address, web_port=8000):
    assert link.startswith('/checkpoints')
    from torch.utils import model_zoo
    from sshtunnel import SSHTunnelForwarder
    with SSHTunnelForwarder(
        (server_address[0], server_address[1]),
        ssh_username=ssh_username,
        ssh_password=ssh_password,
        remote_bind_address=('localhost', web_port)
    ) as server:
        link = f'http://localhost:{server.local_bind_port}'+link.replace('/checkpoints', '')
        print("Downloading with ssh: ", link)
        ckpt = model_zoo.load_url(link)
    return ckpt


def mm_multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments

    Note:
        This function applies the ``func`` to multiple inputs and
            map the multiple outputs of the ``func`` into different
            list. Each list contains the same type of outputs corresponding
            to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results))) 

if __name__ == '__main__':
    inputs_1 = [i for i in range(10)]
    inputs_2 = [i*2 for i in range(10)]
    def fn(x1, x2):
        return x1+x2, x1-x2
    my_result = multi_apply(fn, inputs_1, inputs_2)
    mm_result = mm_multi_apply(fn, inputs_1, inputs_2)
    import pdb; pdb.set_trace()
    for aa,bb in zip(my_result, mm_result):
        for a, b in zip(aa,bb):
            assert a == b, f'{a}, {b} differ'
    print('Success')
    # import pdb; pdb.set_trace()
    # path = 'sample_image/1.png'
    # image = read_img(path)
    # print('Test show from path')
    # show(path)
    # print('Test show from np image')
    i = list(range(10))
    j = list(range(0, 20, 2))
    def fn(a,b):
        return a+b, a-b
    result_1 = multi_apply(fn, i, j)
    reuslt_2 = mm_multi_apply(fn, i, j)

