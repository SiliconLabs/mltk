

if __name__ == '__main__':
    if not __package__:
        import os 
        import sys 
        curdir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.dirname(curdir))
        __package__ = os.path.basename(curdir)# pylint: disable=redefined-builtin

    from .cli.main import main 


    main()