from services.lab_report_reader import analyze_lab_report

ocr_text = """
Haemoglobin
15
13-17

Total Leucocyte Count
5000
4000-10000

Neutrophils
50
40-80

Lymphocytes
40
20-40

Eosinophils
1
1-6

Monocytes
9
2-10

Basophils
0.00
0-1

Absolute Neutrophils
2500.00
2000-7000

Absolute Lymphocytes
2000.00
1000-3000

Absolute Eosinophils
50.00
20-500

Absolute Monocytes
450.00
200-1000

RBCIndices
RBCCount
4.5 -5.5
Mil-
5
lion/cumm

MCV
80.00
81-101

MCH
30.00
27-32

MCHC
37.50
31.5-34.5

Hct
40
40-50

RDW-CV
12
11.6 -14.0

RDW-SD
40
39-46

Platelet Count
300000
150000-410000

PCT
35

MPV
8
7.5-11.5

PDW
9
"""

result = analyze_lab_report(ocr_text)

print("\nFINAL RESULT:")
print(result)