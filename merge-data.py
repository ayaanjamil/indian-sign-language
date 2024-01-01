import csv

fin = open('hand_data_alphabets.csv', 'r')
gin = csv.reader(fin)
next(gin)

fout = open('merged-data.csv', 'w')
gout = csv.writer(fout)

# Corrected column names
header = ["Alphabet"] + [f"Landmark{i}" for i in range(21)]
gout.writerow(header)

prev_frame = None
current_row = None

for i in gin:
    if prev_frame is not None and i[1] != prev_frame:
        gout.writerow(current_row)

    if i[1] != prev_frame:
        current_row = [i[0]]

    current_row.extend([i[7]])

    prev_frame = i[1]

if current_row is not None:
    gout.writerow(current_row)

fin.close()
fout.close()
