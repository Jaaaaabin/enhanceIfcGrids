
import os
from pathlib import Path
from itertools import islice

def get_script_directory():
    """
    Returns the directory where the script is located.
    """
    return os.path.dirname(os.path.abspath(__file__))

def tree(dir_path: Path, level: int=-1, limit_to_directories: bool=True,
         length_limit: int=1000):
    """
    Given a directory Path object, print a visual tree structure while skipping .git and temp* folders.
    """
    # adapted from the source: 
    # https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    
    space = '    '
    branch = '│   '
    tee = '├── '
    last = '└── '

    dir_path = Path(dir_path)  # Accept string coercible to Path
    files = 0
    directories = 0
    
    def should_skip(path: Path): # Skip .git and temp* folders, with possibility to add more.
        return path.name == ".git" or path.name.startswith("temp")
    
    def inner(dir_path: Path, prefix: str='', level=-1):
        nonlocal files, directories
        if not level:
            return  # Stop iterating when level reaches 0
        
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir() and not should_skip(d)] # subject to change
        else:
            contents = [d for d in dir_path.iterdir() if not should_skip(d)] # subject to change
        
        pointers = [tee] * (len(contents) - 1) + [last]  # Adjust visual pointers
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix+extension, level=level-1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1
    
    print(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        print(line)
    
    if next(iterator, None):
        print(f'... length_limit, {length_limit}, reached, counted:')
    
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))

if __name__ == "__main__":
    tree(get_script_directory())