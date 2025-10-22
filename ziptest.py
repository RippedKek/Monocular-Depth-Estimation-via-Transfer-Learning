from zipfile import ZipFile
z=ZipFile('CSVdata.zip')
print([n for n in z.namelist() if n.endswith('nyu2_train.csv')])
print(z.read('data/nyu2_train.csv').decode() .splitlines()[:5])
