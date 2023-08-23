# Logistic Regression From Scratch

### Latar Belakang

Analisis data dan pemodelan statistik telah menjadi bagian integral dari berbagai aspek kehidupan kita saat ini. Salah satu teknik penting dalam analisis data adalah regresi logistik, yang digunakan untuk memprediksi probabilitas kejadian suatu peristiwa dalam suatu kumpulan data. Meskipun ada berbagai pustaka dan perangkat lunak yang menyediakan implementasi regresi logistik, terdapat beberapa alasan penting untuk memahami cara membuatnya dari awal atau "from scratch":

1. Pemahaman yang Mendalam: 

Dalam ilmu data, pemahaman yang mendalam tentang algoritma dan model yang digunakan adalah kunci untuk membuat keputusan yang tepat. Dengan membuat regresi logistik dari awal, seseorang dapat menggali lebih dalam tentang bagaimana model ini benar-benar bekerja.

2. Kustomisasi dan Kontrol: 

Ketika kita membangun model dari awal, kita memiliki kendali penuh terhadap setiap aspeknya. Kustomisasi ini memungkinkan kita untuk menyesuaikan model sesuai dengan kebutuhan spesifik dari data kita. Kita dapat menyesuaikan fungsi kerugian (loss function), penggunaan regularisasi, atau menambahkan fitur-fitur khusus.

3. Pengoptimalan: 

Dalam banyak kasus, model regresi logistik dari pustaka umum mungkin belum dioptimalkan sepenuhnya untuk data kita. Dengan membangunnya sendiri, kita dapat melakukan pengoptimalan yang lebih baik berdasarkan karakteristik data yang spesifik.

4. Pendidikan dan Pembelajaran: 

Untuk orang yang baru belajar tentang analisis data dan machine learning, membangun regresi logistik dari awal adalah cara yang baik untuk memahami konsep dasar seperti fungsi sigmoid, gradien, dan iterasi dalam konteks yang konkret.

5. Implementasi di Lingkungan Tertentu: 

Terkadang, kita mungkin perlu mengimplementasikan model di lingkungan yang tidak memiliki akses ke pustaka machine learning tertentu. Membuat regresi logistik dari awal memungkinkan kita untuk mengintegrasikannya ke dalam lingkungan tersebut.

Dalam rangka membangun regresi logistik dari awal, kita akan membahas aspek-aspek seperti fungsi sigmoid, optimisasi parameter dengan metode gradien, dan bagaimana mengukur kinerja model. Proses ini akan memungkinkan kita untuk lebih memahami inti dari model regresi logistik dan bagaimana menerapkannya pada data yang berbeda.

Dengan demikian, memahami dan membangun regresi logistik dari awal adalah langkah penting dalam perjalanan analisis data dan machine learning, dan itu adalah fondasi yang kuat untuk memahami model yang lebih kompleks di masa depan.

### Kelebihan dan Kekurangan

Logistic Regression adalah salah satu algoritma klasifikasi yang sederhana dan kuat, namun juga memiliki kelebihan dan kekurangan dibandingkan dengan algoritma klasifikasi lainnya. Berikut adalah beberapa kelebihan dan kekurangan utama Logistic Regression dibandingkan dengan algoritma lain:

#### Kelebihan Logistic Regression:

1. Sederhana dan Mudah diinterpretasi: 

Logistic Regression adalah model yang relatif sederhana dan mudah diinterpretasi. Ini membuatnya sangat cocok untuk digunakan dalam situasi di mana interpretasi model adalah faktor kunci.

2. Efisien untuk Data Terstruktur: 

Logistic Regression cenderung efisien ketika digunakan dengan data yang terstruktur, yaitu data yang memiliki fitur-fitur yang jelas dan tidak terlalu kompleks.

3. Bekerja Baik dengan Data Kecil hingga Menengah: 
Logistic Regression dapat bekerja baik dengan jumlah sampel data yang relatif kecil hingga menengah. Ini membuatnya cocok untuk banyak tugas klasifikasi di mana kita memiliki data yang terbatas.

4. Penanganan Overfitting: 
Dengan menggunakan teknik seperti regularisasi (misalnya, L1 atau L2 regularization), Logistic Regression dapat mengatasi masalah overfitting dan meningkatkan generalisasi model.

Output Probabilitas Langsung: Logistic Regression menghasilkan output dalam bentuk probabilitas, yang dapat digunakan untuk membuat keputusan berdasarkan probabilitas relatif dari kelas yang berbeda.

#### Kekurangan Logistic Regression:

1. Tidak Cocok untuk Data Kompleks: 

Logistic Regression cenderung tidak efektif ketika digunakan dengan data yang sangat kompleks atau memiliki pola yang rumit. Ini karena model ini hanya dapat memodelkan hubungan linear antara fitur dan variabel target.

2. Tidak Dapat Menangani Interaksi Tingkat Tinggi: 

Jika ada interaksi tingkat tinggi antara fitur-fitur dalam data, Logistic Regression mungkin tidak mampu menanganinya tanpa pemrosesan tambahan.

3. Sensitif terhadap Outlier: 

Logistic Regression dapat menjadi sensitif terhadap outlier dalam data, yang dapat mempengaruhi hasilnya.

4. Tidak Memiliki Kemampuan Pembelajaran yang Mendalam: 

Logistic Regression adalah model linier sederhana, yang berarti ia tidak memiliki kemampuan pembelajaran yang mendalam seperti algoritma deep learning. Ini dapat membatasi kemampuannya dalam memahami data yang sangat kompleks.

5. Tidak Mengatasi Masalah Multikelas secara Alami: 

Meskipun ada variasi dari Logistic Regression yang dapat menangani masalah multikelas (misalnya, Regresi Logistik Multinomial), ia tidak mengatasi masalah multikelas dengan baik sebagaimana algoritma klasifikasi lain yang dirancang khusus untuk tugas tersebut.

Kesimpulannya, Logistic Regression adalah algoritma klasifikasi yang kuat dan mudah diinterpretasi, tetapi ada batasan yang perlu dipertimbangkan terutama ketika kita berurusan dengan data yang sangat kompleks atau memiliki karakteristik khusus tertentu. Pilihan algoritma tergantung pada jenis data yang Anda miliki dan tujuan analisis Anda.
