## TUBES 2 - K03 - 03 ##
## PEMBAGIAN TUGAS ##
**Konvolusi & Datarange** : 13519165 - Kadek Surya Mahardika
**Sorting**: 13519125 - Habibina Arif Muzayyan


## PERTANYAAN & JAWABAN ## 

**Q: Jelaskan cara kerja program Anda, terutama pada paralelisasi dengan CUDA yang Anda implementasikan!**

A: Untuk operasi convolusi, digunakan 256 threads dan n blok yang setiap bloknya akan melakukan paralelisasi 1 matrix, apabila matriks input lebih besar dari ukuran blok (16x16) maka beberapa thread mungkin saja melakukan lebih dari 1 operasi (looping), untuk algoritma operasi convolusinya sendiri, kami tetap memakai algoritma di file `serial.c` yang diberikan. Untuk operasi datarange, digunakan 256 thread dan n blok (16x16) dimana masing-masing thread bertanggung jawab untuk melakukan operasi datarange 1 matrix, untuk algoritma operasi datarangenya kami tetap memakai algoritma serial yang diberikan. Untuk operasi sorting, kami menggunakan algoritma merge sort yang di paralelisasi dimana pada awalnya akan digunakan 1 blok-1 grid, ketika operasi pemecahan array, masing-masing pemecahan itu akan dihandle oleh 1 blok-1 grid lainnya, ketika operasi penggabungan array oleh block parentnya digunakan algoritma serial yang diberikan.

**Q: Dari waktu eksekusi terbaik program paralel Anda, bandingkan dengan waktu eksekusi program sekuensial yang diberikan. Analisis mengapa waktu eksekusi program Anda bisa lebih lambat / lebih cepat / sama saja. Lalu simpulkan bagaimana CUDA memengaruhi waktu eksekusi program Anda. Buktikan dengan menunjukkan waktu eksekusi yang diperlukan saat demo!**

A: Waktu eksekusi program parallel bisa lebih cepat karena pada dasarnya setiap operasi yang independen yang awalnya dilakukan satu per satu oleh CPU sekarang dieksekusi sekaligus dalam 1 waktu oleh GPU sehingga jelas akan lebih cepat. CUDA disini sangat mempengaruhi waktu eksekusi, karena arsitekturnya yang berdasarkan grid-blok-thread, 1 thread bisa kita anggap sebagai 1 CPU, sehingga dengan banyaknya thread yang diberikan oleh CUDA semakin banyak juga input permasalahan yang bisa dieksekusi.

**Q: Jelaskan secara singkat apakah ada perbedaan antara hasil keluaran program serial dan program paralel Anda, dan jika ada jelaskan juga penyebab dari perbedaan tersebut!**

A: Tidak ada perbedaan, karena setiap input permasalahan sudah dimapping dan dieksekusi dengan benar oleh GPU

**Q: Dengan paralelisasi yang Anda implementasikan, untuk bagian perhitungan konvolusi saja, dari 3 kasus berikut yang manakah yang waktu eksekusinya paling cepat dan mengapa?
(a) Jumlah Matrix: 10000, Ukuran Kernel: 1x1, Ukuran Matrix: 1x1
(b) Jumlah Matrix: 1, Ukuran Kernel: 1x1, Ukuran Matrix: 100x100
(c) Jumlah Matrix: 1, Ukuran Kernel: 100x100, Ukuran Matrix: 100x100**

A: Waktu eksekusi yang tercepat adalah `(a) Jumlah Matrix: 10000, Ukuran Kernel: 1x1, Ukuran Matrix: 1x1`. Alasannya karena ukuran matrixnya yang masih dibawah 16x16 dan CUDA sendiri menyediakan maksumimum sekitar 65 ribu blocks, seperti penjelasan diatas bahwa skema parallelisasi konvolusi adalah 1 block untuk 1 matrix sehingga semua input permasalahan bisa termapping/tereksekusi secara paralalel tanpa adanya perulangan untuk setiap thread.
