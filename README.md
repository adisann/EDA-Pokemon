# EDA-Pokemon

Sering kali ketika memulai bermain pokemon, merasa kebingungan dalam memilih pokemon-pokemon kuat untuk memulai petualangan, bagaimana cara mengatasinya?

<img src="https://github.com/user-attachments/assets/4bad8aab-83b6-48f1-b76e-9f171645d2f8" width="200em">

Inilah pendekatan yang dilakukan untuk memiliki pokemon yang terkuat dalam melakukan petualangan sebagai seorang pokemon Trainer

<img src="https://github.com/user-attachments/assets/d54797d6-da37-4e69-a0cc-8aadc4fe3883" width="200em">


## Pengantar Pokemon EDA
Pada proyek ini akan menggunakan dataset yang tersedia di kaggle: https://www.kaggle.com/datasets/rounakbanik/pokemon/data, dimana dalam dataset hanya ada pokemon sampai generasi ke 7, dan memiliki sekitar 41 kolom, tetapi disini tidak akan menggunakan semua kolom tersebut. Penjelasan dataset yang lebih rinci bisa dilihat pada kaggle di atas.

Penjelasan kolom:

*    name: The English name of the Pokemon
*    japanese_name: The Original Japanese name of the Pokemon
*    pokedex_number: The entry number of the Pokemon in the National Pokedex
*   percentage_male: The percentage of the species that are male. Blank if the Pokemon is genderless.
*   type1: The Primary Type of the Pokemon
*   type2: The Secondary Type of the Pokemon
*   classification: The Classification of the Pokemon as described by the Sun and Moon Pokedex
*   height_m: Height of the Pokemon in metres
*   weight_kg: The Weight of the Pokemon in kilograms
*   capture_rate: Capture Rate of the Pokemon
*   base_egg_steps: The number of steps required to hatch an egg of the Pokemon
*   abilities: A stringified list of abilities that the Pokemon is capable of having
*   experience_growth: The Experience Growth of the Pokemon
*   base_happiness: Base Happiness of the Pokemon
*   against_?: Eighteen features that denote the amount of damage taken against an attack of a particular type
*   hp: The Base HP of the Pokemon
*   attack: The Base Attack of the Pokemon
*   defense: The Base Defense of the Pokemon
*   sp_attack: The Base Special Attack of the Pokemon
*   sp_defense: The Base Special Defense of the Pokemon
*   speed: The Base Speed of the Pokemon
*   generation: The numbered generation which the Pokemon was first introduced
*   is_legendary: Denotes if the Pokemon is legendary.

Dimana, kolom yang digunakan pada visualisasi ini adalah: ["name", "abilities", "classification", "type1", "type2", "attack", "hp", "defense", "speed", "generation", "base_happiness", "capture_rate", "experience_growth", sp_attack", "sp_defense", "is_legendary"]

## Library yang Digunakan

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from math import sqrt

df = pd.read_csv("https://raw.githubusercontent.com/adisann/EDA-Pokemon/refs/heads/main/pokemon.csv")
print(df.head())

```
<img src="https://github.com/user-attachments/assets/bf835cc5-2a1d-4d3a-9508-2948cb490d3f" width="200em">

## Data Cleaning

```python
df.info()
```
<img src="https://github.com/user-attachments/assets/3e69cdcc-f928-488c-9afb-202107239d88" width="200em">


Ada beberapa data yang memiliki null values, yang dilakukan adalah dengan fillna dengan "None", karena ini merupakan Missing Not at Random dikarenakan datanya missing karena tidak semua pokemon memiliki dual type (dua tipe).

```python
df['type2'].fillna('none', inplace = True)
```

Memeriksa kembali apakah ada data yang kosong
```python
df.isnull().values.any()
```
```python
False
```

## Exploratory Data Analysis

### Jumlah Pokemon
Melihat banyaknya pokemon

```python
df.shape
```

```python
(801, 16)
```

```
sns.countplot(x=df['generation'],palette='Set2').set(xlabel='Generasi', ylabel='Jumlah Pokemon')
```

<img src="https://github.com/user-attachments/assets/b57271fb-9fc9-41f6-84bd-5ef67369d536" width="200em">


Ada pattern yang terlihat pada plot diatas, generasi dengan angka ganjil cenderung lebih tinggi dari generasi setelahnya(genap), pada generasi pertama jumlah pokemon lebih tinggi dari generasi ke2, generasi ke3 lebih tinggi dari ke4 dan seterusnya.

*    Generasi ke-5 memiliki jumlah pokemon terbanyak, sedangkan generasi ke-6 memiliki jumlah pokemon paling sedikit
Setelah melihat pembagian pokemon berdasarkan generasi, sekarang saya akan melihat Jumlah pokemon berdasarkan tipe utamanya.

Dalam dunia Pokemon, "tipe" mengacu pada klasifikasi elemen atau sifat yang dimiliki oleh setiap Pokemon. Setiap Pokemon memiliki satu (single-type) atau dua tipe (dual-type) yang mempengaruhi kekuatan, kelemahan, dan resistensi mereka terhadap serangan dan kondisi tertentu.


Menampilkan jumlah setiap tipe pokemon untuk single type dan dual types

```python
# Menghitung jumlah dari tiap tipe utama dan mengurutkan
type_counts = df['type1'].value_counts().reset_index()
type_counts.columns = ['type1', 'count']

# Membuat plot yang diurutkan dari terbesar ke terkecil
plt.figure(figsize=(10, 6))
ax = sns.barplot(y='type1', x='count', data=type_counts, palette='Set2')

# Menambahkan count di ujung bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                ha='center', va='center')

# Mengatur label
ax.set(xlabel='Jumlah Pokemon', ylabel='Tipe Utama Pokemon')
plt.title('Jumlah Pokemon Berdasarkan Tipe Utama', fontsize=16)
plt.show()
```

<img src="https://github.com/user-attachments/assets/fcbae481-6780-4d0c-a0f6-edbb23c1be27" width="200em">

<img src="https://github.com/user-attachments/assets/5b6e7e18-a5ea-4f57-81b9-110aa4fff113" width="200em">


Dimana, setelah dilihat-lihat kembali, bahwa perbandingan antara single type dan dual type itu hampir sama

<img src="https://github.com/user-attachments/assets/001715b9-5c12-4c1d-bb49-a6363c186b3e" width="200em">

### Mencari Pokemon yang Sulit Ditangkap
Mengetahui tipe pokemon yang sulit dan mudah ditangkap

```python
# Mengatur ukuran plotnya
plt.figure(figsize=(17,8))

# Membuat boxplot dengan sumbu x sebagai primary type dan y capture ratenya
sns.boxplot(data=df, x="type1", y="capture_rate",palette="Set2").set(xlabel='Tipe Pokemon', ylabel='Capture Rate')

```

<img src="https://github.com/user-attachments/assets/b7b478f3-cc5b-446f-9009-27382e8c181b" width="200em">

semakin tinggi capture rate maka makin mudah pokemon didapatkan, sebaliknya semakin rendah capture rate maka akan semakin sulit pokemon untuk didapat.

Tipe "dragon" adalah tipe pokemon yang paling sulit untuk ditangkap, disusul dengan tipe "steel" dan "rock"
Tipe "fairy" memiliki nilai median paling tinggi, menunjukkan bahwa pokemon bertipe "fairy" relatif mudah untuk ditangkap
Mencari pokemon yang paling mudah, dan paling sulit ditangkap


Kemudian, ini perbandingan untuk capture rate yang terendah (3), dan yang tertinggi (255)

<img src="https://github.com/user-attachments/assets/a0689198-6860-4b2e-b44a-1c191fc782cd" width="200em">

Bisa dikatakan juga, perbandingannya antara yang sulit dan mudah ditangkap hampir sama, tapi apakah ini perlu dikatakan keuntungan? Apakah pokemon yang mudah ditangkap ini bisa digunakan untuk bertualang? Untuk menjawabnya mari membahas ke overall_stat

### Pokemon Terkuat (overall_stat) dan terlemah (overall_stat)

Overall stat adalah kolom baru yang akan dibuat untuk menampung gabungan beberapa base stat yang meliputi attack, hp, defense, speed, sp_attack, dan sp_defense. Tujuannya sebagai parameter penentu untuk membandingkan total stat dari pokemon.

<img src="https://github.com/user-attachments/assets/939ed14d-aaa7-4362-97d1-255b3275fbdc" width="200em">

Highlight yang merah adalah nilai terendah, dan highlight abu-abu yang tertinggi, artinya:

Tipe "bug" memiliki nilai mode, mean, median terendah menunjukkan kebanyakan pokemon tipe bug memiliki overall stat yang relatif kecil
Tipe "dragon" memiliki nilai mean dan nilai median tertinggi, menunjukkan rata-rata pokemon tersebut memiliki overall_stat yang relatif tinggi

### Pokemon Legendary dan Non-Legendary

<img src="https://github.com/user-attachments/assets/1625f640-b675-4c25-a293-edc56b8f903a" width="200em">

Perbandingan antara pokemon legendary dan non-legendary ini, ternyata sangat berbeda jauh, bahkan tidak sampai 10%.
Muncul pertanyaan-pertanyaan seperti, kenapa hanya sedikit jumlah pokemon legendary? Apakah pokemon legendary merupakan golongan pokemon yang terkuat? Atau apakah ada pokemon non-legendary yang lebih kuat dari pokemon legendary?

<img src="https://github.com/user-attachments/assets/f0a4f8c0-48c4-4b40-84ec-b64735e09131" width="200em">

5 pokemon diatas adalah pokemon yang memiliki overall stat tertinggi, Mewtwo dan Rayquaza memiliki overall stats yang sama yaitu 780, disusul oleh Kyogre dan Groudon sebanyak 770 dan terakhir Arceus sebesar 720.

Sebelum membuat visualisasi untuk membandingkan masing-masing base stat pada 5 pokemon tersebut, mencari tahu apakah kelimanya pokemon legendary?

<img src="https://github.com/user-attachments/assets/faf2d3f5-417a-4424-824b-03140fa363c8" width="200em">

Dan, benar saja semuanya adalah legendary

Ini perbandingan antara setiap stat dari kelima pokemon tersebut

<img src="https://github.com/user-attachments/assets/9ec569e0-b6f2-4152-ad20-9ea9b30af9ed" width="200em">

*   Mewtwo memiliki special attack dan speed tertinggi dari 5 pokemon tersebut
*   Rayquaza memiliki attack dan special attack yang tinggi
*   Kyogre lebih dominan pada special attack dan special defense
*   Groudon lebih dominan pada defense
*   Arceus mempunyai stat yang seimbang pada segala aspek

Kemudian, apakah ada pokemon non-legendary yang memiliki nilai overall_stat yang lebih tinggi daripada pokemon legendary?
Inilah perbandingan antara pokemon Non-Legendary terkuat, dan Legendary terlemah

Non-Legendary

<img src="https://github.com/user-attachments/assets/4aa997e2-0887-4173-ab2f-5cd1736ae874" width="200em">


VS


Legendary

<img src="https://github.com/user-attachments/assets/c669d189-9179-439d-9cad-4ec43d52204e" width="200em">


Ternyata perbedaannya sangat jauh, Tyranitar merupakan pokemon non-legendary terkuat dengan overall stat 700 sedangkan Cosmog merupakan pokemon legendary terlemah dengan overall stat 200

Setelah melihat data diatas, ternyata ada pokemon non legendary yang jauh lebih kuat dari pokemon legendary, lantas muncul pertanyaan lain. "Apakah overall_stat bukan penentu status legendary pokemon? apakah ada perbedaan overall_stat pokemon legendary dengan pokemon non-legendary?

Untuk menjawab pertanyaan tersebut, akan dilakukan uji hipotesis.

### Uji Hipotesis Overall_Stat antara pokemon legendary dan non-legendary

Null Hypothesisnya (H0) claim bahwa tidak ada perbedaan signifikan antara overall stat dari pokemon legendary dan pokemon non-legendary, dan Hypothesis alternatifnya (H1) menunjukkan adanya perbedaan yang signifikan

H0 : rata-rata overall_stat antara pokemon legendary dan non-legendary adalah sama

H1 : rata-rata overall_stat antara pokemon legendary dan non-legendary tidak sama

<img src="https://github.com/user-attachments/assets/2a7d312a-caca-4cdd-a60a-e134f0ebd1e5" width="200em">

```python
# Memfilter dataframe untuk hanya menyimpan pokemon legendary dan non-legendary
legendary = df.query('is_legendary == 1')
ordinary = df.query('is_legendary == 0')

# Mencari mean dari kedua jenis pokemon
legendary_mean = legendary['overall_stat'].mean()
ordinary_mean = ordinary['overall_stat'].mean()

# Mencari standard deviation antara kedua jenis pokemon
legendary_std = legendary['overall_stat'].std()
ordinary_std = ordinary['overall_stat'].std()

# Jumlah pokemon pada data
n_legendary = len(legendary)
n_ordinary = len(ordinary)

# Menghitung t-value
t_value = (legendary_mean - ordinary_mean) / sqrt((legendary_std**2 / n_legendary) + (ordinary_std**2 / n_ordinary))**0.5

# degrees of freedom
defree = n_legendary + n_ordinary - 2

# significance level
alpha = 0.05

# menghitung critical value dari two tailed t-test
critical_value = stats.t.ppf(1 - alpha/2, defree)

# Menghitung p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_value), defree))

# Membandingkan t-value dengan critical valude dan p-value dengan alpha
if abs(t_value) > critical_value or p_value < alpha:
    print("Tolak null hypothesis.")
    print("Ada perbedaan signifikan antara rata-rata dari overall stat Pokemon legendary dan pokemon non-legendary.")
else:
    print("Gagal tolak null hypothesis.")
    print("Tidak ada perbedaan signifikan antara rata-rata dari overall stat Pokemon legendary dan pokemon non-legendary.")
```

```
Tolak null hypothesis.
Ada perbedaan signifikan antara rata-rata dari overall stat Pokemon legendary dan pokemon non-legendary.
```


Distribusi Overall_stat Pokemon

<img src="https://github.com/user-attachments/assets/7256e1a2-5e09-4f30-aa14-c21279807bc7" width="200em">

Sesuai uji hipotesis diatas, "Ada perbedaan signifikan antara rata-rata dari overall stat Pokemon legendary dan pokemon non-legendary". Artinya bahwa pokemon legendary mempunyai overall stat yang berbeda dari pokemon biasa, dan status legendaris artinya merupakan golongan khusus yang berisikan pokemon-pokemon tertentu saja.

Dengan ini, bisa disimpulkan dalam pertandingan pokemon, pokemon legendary akan menjadi pilihan terbaik untuk meminimalisir kekalahan. Namun ada aspek yang harus dipertimbangkan lagi, yaitu capture rate. Saya yakin untuk mendapatkan pokemon legendary bukanlah hal yang mudah, untuk itu saya ingin lihat lagi bagaimana korelasi antara capture rate dengan status legendary pokemon, maupun antara pokemon non-legendary lainnya

### Korelasi Antar Fitur

<img src="https://github.com/user-attachments/assets/95f9baea-c329-41d1-97e2-3d9723928e3d" width="200em">

Sehingga, yang mempengaruhi capture_rate itu adalah overall_stat, begitupula legendary terkolerasi dengan overall_stat. Artinya, semakin sulit menangkap pokemon, bisa diartikan bahwa pokemon itu kemungkinan besar memiliki overall_stat yang tinggi, jadi ketika menangkap pokemon yang memiliki capture rate 255, besar kemungkinan pokemon tersebut tidak bisa digunakan sebagai pokemon utama untuk berpetualang

### Mencari Pokemon Starter Terkuat Setiap Generasi

<img src="https://github.com/user-attachments/assets/22295496-f30b-457c-9a33-e3ff7f869e12" width="200em">

```
===========Gen 1===========
        name  overall_stat
0  Bulbasaur           318


===========Gen 2===========
          name  overall_stat
151  Chikorita           318


===========Gen 3===========
        name  overall_stat
251  Treecko           310
254  Torchic           310
257   Mudkip           310


===========Gen 4===========
        name  overall_stat
386  Turtwig           318


===========Gen 5===========
         name  overall_stat
494     Snivy           308
497     Tepig           308
500  Oshawott           308


===========Gen 6===========
        name  overall_stat
655  Froakie           314


===========Gen 7===========
        name  overall_stat
721   Rowlet           320
724   Litten           320
727  Popplio           320
```


**Pokemon Starter Gen 1 (Kanto)**

<img src="https://github.com/user-attachments/assets/272f3dc2-ac51-430a-8457-c462ba43fb5f" width="200em">

**Pokemon Starter Gen 2 (Johto)**

<img src="https://github.com/user-attachments/assets/24a630bb-fd7b-4fa6-9984-2e6b25bcdec2" width="200em">

**Pokemon Starter Gen 3 (Hoenn)**

<img src="https://github.com/user-attachments/assets/9fd74359-d9c2-4d8b-b72b-750db953e80d" width="200em">

**Pokemon Starter Gen 4 (Sinnoh)**
<img src="https://github.com/user-attachments/assets/c2f5bfca-5ec3-493a-aaad-3f07e9f2f3b3" width="200em">

**Pokemon Starter Gen 5 (Unova)**

<img src="https://github.com/user-attachments/assets/de198677-259e-4a70-9a58-5195bf6f50a6" width="200em">

**Pokemon Starter Gen 6 (Kalos)**

<img src="https://github.com/user-attachments/assets/4f912755-06e5-48e4-a160-43d0f5962864" width="200em">

**Pokemon Starter Gen 7 (Alola)**

<img src="https://github.com/user-attachments/assets/6aaab28b-5f51-40cd-8217-c29be1de791c" width="200em">


## Kesimpulan

Dari berbagai penjabaran diatas, dapat disimpulkan bahwa pokemon memiliki karakteristik unik masing-masing, banyak aspek yang harus dipertimbangkan untuk menentukan mana pokemon yang terbaik.


Berikut beberapa poin kesimpulan yang bisa ditarik :

*    gunakan pokemon Mewtwo dan Rayquaza karena mereka merupakan pokemon dengan overall base stat tertinggi
*    Ada beberapa Pokemon non-legendary yang memiliki overall_stat yang lebih tinggi daripada beberapa Pokemon legendary. Kekuatan relatif sebuah Pokemon tidak hanya ditentukan oleh status legendaris atau non-legendarisnya.
*    Menggunakan pokemon dengan tipe "dragon" merupakan pilihan yang bagus karena memiliki overall_stat yang relatif tinggi. dan hindari menggunakan pokemon tipe "bug" karena kebanyakan pokemon tersebut memiliki overall stat yang relatif kecil
*   Namun apabila ingin memilih pokemon starter, bisa memilih Bulbasaur untuk gen 1, Chikorita untuk gen 2, untuk gen 3 bisa pilih bebas karena overall statnya sama, gen 4 Turtwig, Gen 5 bebas, gen 6 Froakie, gen 7 bebas. Namun, apabila memang ingin memilih secara spesifik stat mana yang ingin diunggulkan bisa melihat radarmap diatas
