# GCRO Socioeconomic Data Dictionary
## Climate-Enhanced Dataset with Categorical Labels

---

## 📊 **Dataset Overview**
- **Name**: GCRO Socioeconomic Climate-Enhanced Dataset
- **Records**: 58,616 individuals
- **Geographic Coverage**: Johannesburg Metropolitan Area, South Africa
- **Temporal Coverage**: 2011-2021 (GCRO Quality of Life Survey waves)
- **Primary Use**: Climate vulnerability and heat exposure analysis

---

## 🏠 **KEY DWELLING TYPE VARIABLES** (Primary Heat Vulnerability Indicators)

### **DwellingType** & **dwelling_type_enhanced**
Primary heat vulnerability classification:
- **1.0**: `Formal dwelling (house/flat)` - **12,894 records** - Brick/concrete houses, apartments, townhouses
- **2.0**: `Semi-formal dwelling (backyard room)` - **1,685 records** - Rooms in backyard, separate from main house
- **3.0**: `Informal dwelling (shack/squatter)` - **421 records** - **HIGHEST HEAT VULNERABILITY** - Shacks in settlements

### **A3_dwelling_recode** (Detailed Housing Categories)
- **Formal**: `Formal dwelling (brick/concrete house or flat)` - **11,603 records**
- **Informal**: `Informal dwelling (shack in backyard or settlement)` - **2,694 records**
- **Other**: `Other dwelling type (hostel, traditional, etc.)` - **703 records**

---

## 👥 **DEMOGRAPHIC VARIABLES**

### **Sex**
- **1.0**: `Male` - **6,619 records**
- **2.0**: `Female` - **8,381 records**

### **Race**
- **1.0**: `African/Black` - **12,186 records** (81.2%)
- **2.0**: `Coloured` - **448 records**
- **3.0**: `Indian/Asian` - **514 records**
- **4.0**: `White` - **1,852 records**
- **5.0**: `Other` - **0 records**

### **Age Groups** (Q15_02_age_recode)
Heat vulnerability by age:
- **1.0**: `18-24 years (young adult)` - Moderate heat vulnerability
- **2.0**: `25-34 years (young adult)` - Low heat vulnerability
- **3.0**: `35-44 years (middle-aged)` - Low heat vulnerability
- **4.0**: `45-54 years (middle-aged)` - Moderate heat vulnerability
- **5.0**: `55-64 years (older adult)` - High heat vulnerability
- **6.0**: `65+ years (elderly)` - **HIGHEST HEAT VULNERABILITY**

---

## 💰 **ECONOMIC VARIABLES** (Economic Vulnerability Indicators)

### **EmploymentStatus**
Economic capacity for heat adaptation:
- **1.0**: `Employed full-time` - **4,038 records** - Highest adaptive capacity
- **2.0**: `Employed part-time` - **829 records** - Moderate adaptive capacity
- **3.0**: `Unemployed seeking work` - **5,456 records** - Low adaptive capacity
- **4.0**: `Unemployed not seeking work` - **4,559 records** - Low adaptive capacity
- **5.0**: `Student` - **0 records**
- **6.0**: `Retired` - **0 records**
- **7.0**: `Unable to work (disabled/ill)` - **0 records**

### **Q15_20_income** (Monthly Household Income)
Economic heat vulnerability:
- **1.0**: `No income` - **HIGHEST VULNERABILITY**
- **2.0**: `R1-R800 per month (very low income)` - **VERY HIGH VULNERABILITY**
- **3.0**: `R801-R1600 per month (low income)` - **HIGH VULNERABILITY**
- **4.0**: `R1601-R3200 per month (lower-middle income)` - **MODERATE VULNERABILITY**
- **5.0**: `R3201-R6400 per month (middle income)` - **LOW VULNERABILITY**
- **6.0**: `R6401-R12800 per month (upper-middle income)` - **LOW VULNERABILITY**
- **7.0**: `R12801-R25600 per month (high income)` - **VERY LOW VULNERABILITY**
- **8.0**: `R25601+ per month (very high income)` - **LOWEST VULNERABILITY**

---

## 🎓 **EDUCATION VARIABLES** (Adaptive Capacity Indicators)

### **Education** & **std_education**
Educational attainment affecting heat adaptation knowledge:
- **0.0**: `No formal education` - **1,397 records** - Lowest adaptive capacity
- **1.0**: `Primary education (Grades 1-7)` - **1,216 records** - Low adaptive capacity
- **2.0**: `Secondary education (Grades 8-12)` - **5,044 records** - Moderate adaptive capacity
- **3.0**: `Matric/Grade 12 completed` - **4,421 records** - Good adaptive capacity
- **4.0**: `Post-secondary education (tertiary)` - **2,405 records** - High adaptive capacity
- **5.0**: `University degree or higher` - **0 records** - Highest adaptive capacity

---

## 🏡 **HOUSING QUALITY VARIABLES**

### **Q2_01_dwelling** (Dwelling Satisfaction)
Housing quality indicator for heat protection:
- **Very satisfied**: `Very satisfied with dwelling` - **2,004 records** - Best housing quality
- **Satisfied**: `Satisfied with dwelling` - **7,139 records** - Good housing quality
- **Neither satisfied nor dissatisfied**: `Neutral about dwelling` - **2,757 records** - Moderate quality
- **Dissatisfied**: `Dissatisfied with dwelling` - **3,100 records** - Poor housing quality
- **Very dissatisfied**: `Very dissatisfied with dwelling` - **0 records** - Worst housing quality

### **Q2_14_Drainage** (Infrastructure Quality)
Drainage affects flood risk and heat island effects:
- **1.0**: `Excellent drainage` - Best infrastructure
- **2.0**: `Good drainage` - Good infrastructure
- **3.0**: `Fair drainage` - Moderate infrastructure
- **4.0**: `Poor drainage` - Poor infrastructure
- **5.0**: `No drainage system` - **HIGHEST FLOOD/HEAT RISK**

---

## 🌡️ **HEAT VULNERABILITY INDEX** (Composite Indicator)

### **heat_vulnerability_index** (Scale: 1-5)
Composite heat vulnerability score combining:
- **40% Dwelling Type** (housing quality/type)
- **25% Income Level** (economic adaptive capacity)
- **20% Age Group** (physiological vulnerability)
- **15% Education Level** (knowledge/adaptive capacity)

**Current Distribution**:
- **Low vulnerability (1-2)**: 14,593 records (97.3%)
- **Moderate vulnerability (2-3)**: 407 records (2.7%)
- **High vulnerability (3-4)**: 0 records (0.0%)
- **Very high vulnerability (4-5)**: 0 records (0.0%)

### **heat_vulnerability_category**
Categorical classification:
- `Low vulnerability` - Formal housing, higher income, education
- `Moderate vulnerability` - Mixed conditions
- `High vulnerability` - Informal housing, low income
- `Very high vulnerability` - Multiple risk factors

---

## 📅 **TEMPORAL VARIABLES**

### **survey_wave**
Quality of Life Survey periods:
- `Wave_2011`: **2011 Quality of Life Survey** - **15,000 records**
- `Wave_2013`: **2013 Quality of Life Survey**
- `Wave_2015`: **2015 Quality of Life Survey**
- `Wave_2017`: **2017 Quality of Life Survey**
- `Wave_2019`: **2019 Quality of Life Survey**
- `Wave_2021`: **2021 Quality of Life Survey** - **13,616 records**

---

## 🎯 **CLIMATE ANALYSIS USAGE**

### **Primary Heat Vulnerability Indicators**:
1. **dwelling_type_enhanced** - Direct housing quality assessment
2. **heat_vulnerability_index** - Composite vulnerability score
3. **Q15_20_income** - Economic adaptive capacity
4. **Q15_02_age_recode** - Age-based physiological vulnerability

### **Analysis Recommendations**:
- Use **dwelling_type_enhanced** for direct heat exposure analysis
- Use **heat_vulnerability_index** for overall vulnerability modeling
- Stratify by **income** and **education** for social vulnerability analysis
- Consider **survey_wave** for temporal trend analysis

### **Highest Vulnerability Groups**:
- **Informal dwelling residents** (Type 3.0) - 421 people
- **No income households** - Economic vulnerability
- **Elderly residents (65+)** - Physiological vulnerability
- **Very dissatisfied with dwelling** - Poor housing quality

---

## 📊 **Data Quality Summary**
- **Geographic Coverage**: 100% geocoded to Johannesburg wards
- **Dwelling Type Coverage**: 25.6% of total dataset (15,000/58,616)
- **Demographics Coverage**: 25.6% complete
- **Economic Indicators**: Variable coverage by survey wave
- **Temporal Consistency**: Consistent methodology across waves

**Note**: The dataset contains both coded values (1.0, 2.0, etc.) and descriptive labels (_label columns) for all categorical variables.