
def export_aspect(domain, data_dir):
    aspect_list = list()
    
    fa = codecs.open('../dictionary/' + domain + '_aspect.txt', 'w', 'utf-8')
    for file in os.listdir(data_dir):
        if not (file.endswith('.txt') and domain in file):
            continue
            
        f = codecs.open(data_dir + file, 'r', 'utf-8')
        for line in f:
            for word in line.split(' '):
                if '{as' in word:
                    aspect_list.append(word.split('{')[0].strip())
        f.close()
            
    for w in sorted(set(aspect_list)):
        fa.write(w + '\n')
    
    fa.close()
    
    return set(aspect_list)