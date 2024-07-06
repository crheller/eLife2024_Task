import os
def results_file(directory, site, batch, modelname, filename):
    """
    Define the output results file
    """
    batch = str(batch)
    if os.path.isdir(directory):
        pass
    else:
        os.mkdir(directory)
    
    batchdir = os.path.join(directory, batch)
    if os.path.isdir(batchdir):
        pass
    else:
        os.mkdir(batchdir)

    sitedir = os.path.join(directory, batch, site)
    if os.path.isdir(sitedir):
        pass
    else:
        os.mkdir(sitedir)

    modeldir = os.path.join(directory, batch, site, modelname)
    if os.path.isdir(modeldir):
        pass
    else:
        os.mkdir(modeldir)

    filepath = os.path.join(directory, batch, site, modelname, filename)
    return filepath


def local_results_file(directory, site, modelname, filename):
    """
    Define the output results file
    """
    if os.path.isdir(directory):
        pass
    else:
        os.mkdir(directory)

    sitedir = os.path.join(directory, site)
    if os.path.isdir(sitedir):
        pass
    else:
        os.mkdir(sitedir)

    modeldir = os.path.join(directory, site, modelname)
    if os.path.isdir(modeldir):
        pass
    else:
        os.mkdir(modeldir)

    filepath = os.path.join(directory, site, modelname, filename)
    return filepath