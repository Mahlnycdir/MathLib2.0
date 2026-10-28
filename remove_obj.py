import shutil
import os 
import sys


def print_progress(done, total, width=20):
    filled = int(width * done / total)
    percent = 100 * done / total
    sys.stdout.write("\r[%-*s] %3d%%" % (width, "=" * filled, percent))
    sys.stdout.flush()


def remove_obj(dir_path = None, 
               dir_to_remove = None, 
               exceptions = [], 
               type = "dir"):
    """
    Removes a file/directory in the given path and in all other subdirectories. If no path is 
    given, the current directory of the script is considered. 
    
    Args:
        dir_path (str, optional): Path of the main-directory. Defaults to None.
        dir_to_remove (str, optional): The name of the target. Defaults to None.
        exceptions (list, optional): A list of directories, skiped by the script i.e. [".venv", "bin"]. Defaults to [].
        type (str, optional): Type of the object (directory or file). Defaults to "dir".

    Raises:
        ValueError: If no `dir_to_remove` is of NoneType 
        TypeError: If `dir_to_remove` is no string
    Returns:
        list : `paths_to_remove`
    """
    if dir_to_remove is None:
        raise ValueError("An entry for `dir_to_remove` is needed!")
    if not isinstance(dir_to_remove, str):
        raise TypeError("`dir_to_remove` needs to be a string!")
    
    if dir_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    paths_to_remove = []
    
    objects = os.scandir(dir_path)
    
    for entry in objects:
        if not entry.is_dir():
            continue
        try:
            exceptions.index(entry.name)
        except ValueError:
            if entry.name == dir_to_remove:
                paths_to_remove.append(dir_path + "/" + dir_to_remove + "/")
            else:
                for path in remove_obj(dir_path = entry.path, dir_to_remove = dir_to_remove, type=type):
                    try:
                        paths_to_remove.index(path)
                    except ValueError: 
                        paths_to_remove.append(path)
        
    total = len(paths_to_remove)
            
    for i,path in enumerate(paths_to_remove, start=1):
        shutil.rmtree(path, ignore_errors=False)
        print_progress(i, total)
        try:
            paths_to_remove.remove(path)
        except ValueError:
            continue
    
    return paths_to_remove


if __name__ == "__main__":
    remove_obj(dir_to_remove="__pycache__")