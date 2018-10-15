from zipfile import ZipFile


def _read_line(line):
    values = []
    for value in line.split(','):
        values.append(float(value.strip()))
    return values


def _read_file(fi):
    dat = []
    for line in fi:
        try:
            dat.append(_read_line(line.decode()))
        except:
            pass
    return dat


class Data(object):
    """Encapsulates the zip file containing the trace data with a simple API """
    zip = None

    def __init__(self, data_file_path):
        self.zip = ZipFile(data_file_path)

    def get_labels(self):
        return [name.split('_')[1] for name in self.zip.namelist()]

    def get_instances(self, target_label=None):
        X = []
        for filename in self.zip.namelist():
            _, label = filename.split('_')
            if target_label is None or target_label == label:
                with self.zip.open(filename, 'rU') as fi:
                    X.extend(_read_file(fi))
        return X
