import re
import pytesseract
from datetime import datetime

list = []
# imag = 'Crop/crop_1657706659987.jpg'
imag = 'Crop/crop_1657707301368.jpg'

#OCR }}}}}}}}}}}}}}}}
p = pytesseract.image_to_string(imag, lang='eng')

#extracted text without new line }}}}}}}}}}}}}
p = p.replace("\n", " ")
# print("ID_CARD_OCR_:\n\n", p)
# print(type(p))
# pocr = "".join(p.split())
# for i in p:
#     ocr_line.append(i)
# print("char_list",ocr_line)

# extract Gr.No. }}}}}}}}}}}}}}}}}
r = re.search(r'\d+', p).group()
# print("Extracted Text from ID card: \n\n Gr. No. ",r)

# to cut Gr.No.}}}}}}}}}}}}}}
# s = p.replace(r, "")
# print("\ncut----", s, type(r))
# res = p.split()[0]
# print("\nsplit'''''''''",res)

# The Regex pattern to match al characters on and before '-'
pattern  = ".*" + r 
# Remove all characters before Gr.No. from string
str = re.sub(pattern, '', p)
# print("\nstr_____", str)


res = str[0:30]
nam=re.sub('[a-z]', "", res)
# print("\nFull Name: ",nam)
# The Regex pattern to match al characters on and before '-'
pattern  = ".*" + res 
# Remove all characters before name from string
sr = re.sub(pattern, '', str)
# print("\nsr_____", sr)

a = sr.split(",")
# print("address start,,,,,,,,,,,", a)

grno = list.append(r)
#remove end (rear) space from str}}}}}}}}}}
# name = list.append(res.rstrip())
#remove start-end space from str}}}}}}}}}}
name = list.append(" ".join(nam.split()))

mono = sr.split(":")[1]
mb = mono.split("/")[0]
mbi = mono.split("/")[1]
# print("mono.....",mono, '\n', mbi)
mn = re.search(r'\d+', mbi).group()
# print("Mo.No. :",mb,"/", mn)
mi = mb+"/"+mn
mo_no = list.append(" ".join(mi.split()))

# d = re.findall(r'\d', mbi)
print('date',d)

match_dt = re.search(r'\d{2}-\d{2}-\d{4}', mbi)
  # feeding format
dateStr = datetime.strptime(match_dt.group(), '%d-%m-%Y').date()
dt = dateStr.strftime("%d-%m-%Y")
# print("Computed date : ",type(dt), dt)
# st = str(dt)
# print("str", st)
dob = list.append(dt)

crc = ['M.Ed.', 'MEd.', 'M.Ed', 'med', 'mEd', 'm.ed.', 'BCA', 'bca', 'B.Ed.', 'bed', 'BED', 'Bcom', 'BCOM', 'B.COM.', 'B.com', 'B.com.', 'bcom', 'b.com', 'b.com.', 'bba', 'BBA', 'BSC', 'bsc', 'Bsc', 'bSC', 'bsC', 'bSc',
    'msc-chem', 'MSCchem', 'MSC', 'Msc', 'MSc', 'msc', 'MsC', 'MSCCHEM', 'Msc-chem', 'Msc-Chem',  'msc-it', 'MSCit', 'MSC-it', 'MscIT', 'MSc-IT', 'msc-IT', 'MSC-IT', 'Msc-it', 'MSc-it']

cor = any(c in mbi for c in crc)
# print("course in list of crc:::::::::", cor)
c = [word for word in crc if word in mbi]
# print("Course: ", type(c))
course = list.append(" ".join(c))
# if cor:
#    for j in mbi.split():
        # print(j) 
    
match_y = re.findall(r'\d{4}-\d{4}', mbi)
# print("Passing year: ",match_y)
passy = list.append(" ".join(match_y))

# print("Data::::::::::::::", list)