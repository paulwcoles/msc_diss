import re
import os
import shutil

def get_ctm_paths(top_dir):
    directories = [top_dir + name + '/rescore-dnn-seq-mfcc/out/ctm/' for name in \
    os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, name)) and name.startswith('2')]

    ctm_paths = []
    for directory in directories:
        try:
            print "Transcript retrieved from %s" % directory
            ctm_paths.append(directory + str(os.listdir(directory)[0]))
        except:
            print "Missing transcript in %s" % directory
            continue
    return ctm_paths

def copy_ctm(ctm, target):
    print "Copying file %s" % ctm
    shutil.copy(ctm, target)

def parse_raw_files():
    for f in os.listdir(data_dir):
        print "Parsing raw file %s" % str(f)
        lines = []
        if f != '.DS_Store':
            print "Parsing file %s..." % f
            with open(data_dir + f, 'r') as transcript:
                read_count = 0
                for line in transcript:
                    split_line = line.split()
                    utt_id = split_line[0]
                    word = split_line[4]
                    lines.insert(read_count, [utt_id, word])
                    read_count += 1
            shortname = os.path.splitext(f)[0]
            transcripts[shortname] = lines
    print str(read_count) + " lines parsed.\n"

def write_text_files():
    for key, value in transcripts.iteritems():
        with open(output_dir + key + '_parsed.txt', 'w') as outfile:
            print "Writing output for file %s..." % key
            write_count = 0
            for line in value:
                if write_count == 0 or value[write_count][0] == value[write_count - 1][0]:
                    outfile.write(line[1] + ' ')
                else:
                    outfile.write('\n\n' + line[1] + ' ')
                write_count += 1


if __name__ == "__main__":
    top_dir = '/disk/data3/r4today/'
    ctm_paths = get_ctm_paths(top_dir)
    target = './data/today/transcripts/'
    data_dir = './data/today/transcripts/'
    output_dir = './data/today/parsed/'
    transcripts = {}
    print ''
    for ctm in ctm_paths:
        try:
            copy_ctm(ctm, target)
        except:
            continue
    print ''
    parse_raw_files()
    write_text_files()
