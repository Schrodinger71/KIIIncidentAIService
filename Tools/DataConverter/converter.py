import openpyxl

dataframe = openpyxl.load_workbook("convert.xlsx")
data = dataframe.active

convertedWb = openpyxl.Workbook()

convSt = convertedWb.active

convSt.append(["Gender",  # Пол
               "Hormonal Background",  # Гормональный фон
               "The cardiovascular system",  # Сердечно сосудистая система
               "Genitourinary system",  # Мочеполовая система
               "Gastrointestinal tract",  # Желудочно- кишечный тракт
               "Angiopathy",  # Ангиопатия
               "Tumors of other localization",  # Опухоли другой локализации
              "RBC",
               "MCV",
               "RDW",
               "RDWa",
               "HCT",
               "PLT",
               "MPV",
               "PDW",
               "PCT",
               "LPCR",
               "WBC",
               "HGB",
               "MCH",
               "MCHC",
               "LYM",
               "GRAN",
               "MID",
               "LIM%",
               "GRA%",
               "MID%",
               "NEUT",
               "BO",
               "BASO",
               "MON",
               "Ultrasound"]  # УЗИ
              )

genderData = data['A4':'B123']

for index, row in enumerate(genderData):
    for col in row:
        if col.column_letter == 'A' and col.value == '+':
            convSt[f'A{index+2}'] = -1
        elif col.column_letter == 'B' and col.value == '+':
            convSt[f'A{index+2}'] = 1

hormonalBackgroundData = data['C4':'E123']

for index, row in enumerate(hormonalBackgroundData):
    for col in row:
        if col.column_letter == 'C' and col.value == '+':
            convSt[f'B{index+2}'] = 0
        elif col.column_letter == 'D' and col.value == '+':
            convSt[f'B{index+2}'] = -1
        elif col.column_letter == 'E' and col.value == '+':
            convSt[f'B{index+2}'] = 1

cardioSystemData = data['F4':'F123']

for index, row in enumerate(cardioSystemData):
    for col in row:
        if col.column_letter == 'F' and col.value == '+':
            convSt[f'C{index+2}'] = 1
        else:
            convSt[f'C{index+2}'] = 0

genitourinartData = data['G4':'G123']

for index, row in enumerate(genitourinartData):
    for col in row:
        if col.column_letter == 'G' and col.value == '+':
            convSt[f'D{index+2}'] = 1
        else:
            convSt[f'D{index+2}'] = 0

gastrointestinalData = data['H4':'H123']

for index, row in enumerate(gastrointestinalData):
    for col in row:
        if col.column_letter == 'H' and col.value == '+':
            convSt[f'E{index+2}'] = 1
        else:
            convSt[f'E{index+2}'] = 0

angiopathyData = data['I4':'I123']

for index, row in enumerate(angiopathyData):
    for col in row:
        if col.column_letter == 'I' and col.value == '+':
            convSt[f'F{index+2}'] = 1
        else:
            convSt[f'F{index+2}'] = 0

tumorsData = data['J4':'J123']

for index, row in enumerate(tumorsData):
    for col in row:
        if col.column_letter == 'J' and col.value == '+':
            convSt[f'G{index+2}'] = 1
        else:
            convSt[f'G{index+2}'] = 0

RBCData = data['K4':'M123']

for index, row in enumerate(RBCData):
    for col in row:
        if col.column_letter == 'K' and col.value == '+':
            convSt[f'H{index+2}'] = -1
        elif col.column_letter == 'L' and col.value == '+':
            convSt[f'H{index+2}'] = 0
        elif col.column_letter == 'M' and col.value == '+':
            convSt[f'H{index+2}'] = 1

MCVData = data['N4':'P123']

for index, row in enumerate(MCVData):
    for col in row:
        if col.column_letter == 'N' and col.value == '+':
            convSt[f'I{index+2}'] = -1
        elif col.column_letter == 'O' and col.value == '+':
            convSt[f'I{index+2}'] = 0
        elif col.column_letter == 'P' and col.value == '+':
            convSt[f'I{index+2}'] = 1

RDWData = data['Q4':'S123']

for index, row in enumerate(RDWData):
    for col in row:
        if col.column_letter == 'Q' and col.value == '+':
            convSt[f'J{index+2}'] = -1
        elif col.column_letter == 'R' and col.value == '+':
            convSt[f'J{index+2}'] = 0
        elif col.column_letter == 'S' and col.value == '+':
            convSt[f'J{index+2}'] = 1

RDWaData = data['T4':'V123']

for index, row in enumerate(RDWaData):
    for col in row:
        if col.column_letter == 'T' and col.value == '+':
            convSt[f'K{index+2}'] = -1
        elif col.column_letter == 'U' and col.value == '+':
            convSt[f'K{index+2}'] = 0
        elif col.column_letter == 'V' and col.value == '+':
            convSt[f'K{index+2}'] = 1

HCTData = data['W4':'Y123']

for index, row in enumerate(HCTData):
    for col in row:
        if col.column_letter == 'W' and col.value == '+':
            convSt[f'L{index+2}'] = -1
        elif col.column_letter == 'X' and col.value == '+':
            convSt[f'L{index+2}'] = 0
        elif col.column_letter == 'Y' and col.value == '+':
            convSt[f'L{index+2}'] = 1

PLTData = data['Z4':'AB123']

for index, row in enumerate(PLTData):
    for col in row:
        if col.column_letter == 'Z' and col.value == '+':
            convSt[f'M{index+2}'] = -1
        elif col.column_letter == 'AA' and col.value == '+':
            convSt[f'M{index+2}'] = 0
        elif col.column_letter == 'AB' and col.value == '+':
            convSt[f'M{index+2}'] = 1

MPVData = data['AC4':'AE123']

for index, row in enumerate(MPVData):
    for col in row:
        if col.column_letter == 'AC' and col.value == '+':
            convSt[f'N{index+2}'] = -1
        elif col.column_letter == 'AD' and col.value == '+':
            convSt[f'N{index+2}'] = 0
        elif col.column_letter == 'AE' and col.value == '+':
            convSt[f'N{index+2}'] = 1

PDWData = data['AF4':'AH123']

for index, row in enumerate(PDWData):
    for col in row:
        if col.column_letter == 'AF' and col.value == '+':
            convSt[f'O{index+2}'] = -1
        elif col.column_letter == 'AG' and col.value == '+':
            convSt[f'O{index+2}'] = 0
        elif col.column_letter == 'AH' and col.value == '+':
            convSt[f'O{index+2}'] = 1

PCTData = data['AI4':'AK123']

for index, row in enumerate(PCTData):
    for col in row:
        if col.column_letter == 'AI' and col.value == '+':
            convSt[f'P{index+2}'] = -1
        elif col.column_letter == 'AJ' and col.value == '+':
            convSt[f'P{index+2}'] = 0
        elif col.column_letter == 'AK' and col.value == '+':
            convSt[f'P{index+2}'] = 1

LPCRData = data['AL4':'AN123']

for index, row in enumerate(LPCRData):
    for col in row:
        if col.column_letter == 'AL' and col.value == '+':
            convSt[f'Q{index+2}'] = -1
        elif col.column_letter == 'AM' and col.value == '+':
            convSt[f'Q{index+2}'] = 0
        elif col.column_letter == 'AN' and col.value == '+':
            convSt[f'Q{index+2}'] = 1

WBCData = data['AO4':'AQ123']

for index, row in enumerate(WBCData):
    for col in row:
        if col.column_letter == 'AO' and col.value == '+':
            convSt[f'R{index+2}'] = -1
        elif col.column_letter == 'AP' and col.value == '+':
            convSt[f'R{index+2}'] = 0
        elif col.column_letter == 'AQ' and col.value == '+':
            convSt[f'R{index+2}'] = 1

HGBData = data['AR4':'AT123']

for index, row in enumerate(HGBData):
    for col in row:
        if col.column_letter == 'AR' and col.value == '+':
            convSt[f'S{index+2}'] = -1
        elif col.column_letter == 'AS' and col.value == '+':
            convSt[f'S{index+2}'] = 0
        elif col.column_letter == 'AT' and col.value == '+':
            convSt[f'S{index+2}'] = 1

MCHData = data['AU4':'AW123']

for index, row in enumerate(MCHData):
    for col in row:
        if col.column_letter == 'AU' and col.value == '+':
            convSt[f'T{index+2}'] = -1
        elif col.column_letter == 'AV' and col.value == '+':
            convSt[f'T{index+2}'] = 0
        elif col.column_letter == 'AW' and col.value == '+':
            convSt[f'T{index+2}'] = 1

MCHCData = data['AX4':'AZ123']

for index, row in enumerate(MCHCData):
    for col in row:
        if col.column_letter == 'AX' and col.value == '+':
            convSt[f'U{index+2}'] = -1
        elif col.column_letter == 'AY' and col.value == '+':
            convSt[f'U{index+2}'] = 0
        elif col.column_letter == 'AZ' and col.value == '+':
            convSt[f'U{index+2}'] = 1

LYMData = data['BA4':'BC123']

for index, row in enumerate(LYMData):
    for col in row:
        if col.column_letter == 'BA' and col.value == '+':
            convSt[f'V{index+2}'] = -1
        elif col.column_letter == 'BB' and col.value == '+':
            convSt[f'V{index+2}'] = 0
        elif col.column_letter == 'BC' and col.value == '+':
            convSt[f'V{index+2}'] = 1

GRANData = data['BD4':'BF123']

for index, row in enumerate(GRANData):
    for col in row:
        if col.column_letter == 'BD' and col.value == '+':
            convSt[f'W{index+2}'] = -1
        elif col.column_letter == 'BE' and col.value == '+':
            convSt[f'W{index+2}'] = 0
        elif col.column_letter == 'BF' and col.value == '+':
            convSt[f'W{index+2}'] = 1

MIDData = data['BG4':'BI123']

for index, row in enumerate(MIDData):
    for col in row:
        if col.column_letter == 'BG' and col.value == '+':
            convSt[f'X{index+2}'] = -1
        elif col.column_letter == 'BH' and col.value == '+':
            convSt[f'X{index+2}'] = 0
        elif col.column_letter == 'BI' and col.value == '+':
            convSt[f'X{index+2}'] = 1

LIMProcData = data['BJ4':'BL123']

for index, row in enumerate(LIMProcData):
    for col in row:
        if col.column_letter == 'BJ' and col.value == '+':
            convSt[f'Y{index+2}'] = -1
        elif col.column_letter == 'BK' and col.value == '+':
            convSt[f'Y{index+2}'] = 0
        elif col.column_letter == 'BL' and col.value == '+':
            convSt[f'Y{index+2}'] = 1

GRAProcData = data['BM4':'BO123']

for index, row in enumerate(GRAProcData):
    for col in row:
        if col.column_letter == 'BM' and col.value == '+':
            convSt[f'Z{index+2}'] = -1
        elif col.column_letter == 'BN' and col.value == '+':
            convSt[f'Z{index+2}'] = 0
        elif col.column_letter == 'BO' and col.value == '+':
            convSt[f'Z{index+2}'] = 1

MIDProcData = data['BP4':'BR123']

for index, row in enumerate(MIDProcData):
    for col in row:
        if col.column_letter == 'BP' and col.value == '+':
            convSt[f'AA{index+2}'] = -1
        elif col.column_letter == 'BQ' and col.value == '+':
            convSt[f'AA{index+2}'] = 0
        elif col.column_letter == 'BR' and col.value == '+':
            convSt[f'AA{index+2}'] = 1

NEUTData = data['BS4':'BU123']

for index, row in enumerate(NEUTData):
    for col in row:
        if col.column_letter == 'BS' and col.value == '+':
            convSt[f'AB{index+2}'] = -1
        elif col.column_letter == 'BT' and col.value == '+':
            convSt[f'AB{index+2}'] = 0
        elif col.column_letter == 'BU' and col.value == '+':
            convSt[f'AB{index+2}'] = 1

EOData = data['BV4':'BX123']

for index, row in enumerate(EOData):
    for col in row:
        if col.column_letter == 'BV' and col.value == '+':
            convSt[f'AC{index+2}'] = -1
        elif col.column_letter == 'BW' and col.value == '+':
            convSt[f'AC{index+2}'] = 0
        elif col.column_letter == 'BX' and col.value == '+':
            convSt[f'AC{index+2}'] = 1

BASOData = data['BY4':'CA123']

for index, row in enumerate(BASOData):
    for col in row:
        if col.column_letter == 'BY' and col.value == '+':
            convSt[f'AD{index+2}'] = -1
        elif col.column_letter == 'BZ' and col.value == '+':
            convSt[f'AD{index+2}'] = 0
        elif col.column_letter == 'CA' and col.value == '+':
            convSt[f'AD{index+2}'] = 1

MONData = data['CB4':'CD123']

for index, row in enumerate(MONData):
    for col in row:
        if col.column_letter == 'CB' and col.value == '+':
            convSt[f'AE{index+2}'] = -1
        elif col.column_letter == 'CC' and col.value == '+':
            convSt[f'AE{index+2}'] = 0
        elif col.column_letter == 'CD' and col.value == '+':
            convSt[f'AE{index+2}'] = 1

UltraSoundData = data['CE4':'CF123']

for index, row in enumerate(UltraSoundData):
    for col in row:
        if col.column_letter == 'CE' and col.value == '+':
            convSt[f'AF{index+2}'] = 0
        elif col.column_letter == 'CF' and col.value == '+':
            convSt[f'AF{index+2}'] = 1

convertedWb.save('Converted.xlsx')
