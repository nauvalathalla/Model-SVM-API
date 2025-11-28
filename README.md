# Model-SVM-API

# 🐛 IoT Anomaly Detection System (One-Class SVM)
Sistem ini mendeteksi anomali dari data sensor (mis. suhu, kelembapan, pH, oksigen, turbidity) menggunakan model **One-Class SVM (OCSVM)**. Proyek ini dirancang untuk integrasi dengan **IoT** dan **Aplikasi Android**, terutama untuk monitoring maggot farm, biopond, atau sistem lingkungan lainnya.

---

## 🚀 Fitur Utama
- Training model OCSVM menggunakan data dummy.
- Prediksi status sensor: `normal` atau `anomaly`.
- Skor confidence dari model.
- API berbasis **FastAPI**.
- Endpoint siap pakai untuk Android (Kotlin + Retrofit).
- Mudah di-deploy ke server lokal atau cloud.

---

## 📁 Struktur Project
.
├── ocsvm_anomaly_fastapi.py
├── ocsvm_maggot.pkl # Hasil model
├── README.md
└── requirements.txt

yaml
Copy code

---

## 🔧 Instalasi & Menjalankan Server

### 1. Clone repository
```bash
git clone <repository-url>
cd anomaly-detection
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Jalankan FastAPI server
bash
Copy code
uvicorn ocsvm_anomaly_fastapi:app --reload
4. Buka Dokumentasi API
arduino
Copy code
http://127.0.0.1:8000/docs
📡 API Documentation
▶️ POST /predict
Prediksi status data sensor.

Request Body
json
Copy code
{
  "temperature": 31.2,
  "ph": 7.5,
  "turbidity": 2.1,
  "oxygen": 4.8
}
Response
Normal:

json
Copy code
{
  "status": "normal",
  "score": 0.12,
  "timestamp": "2025-11-28T02:15:00Z"
}
Anomaly:

json
Copy code
{
  "status": "anomaly",
  "score": -0.55,
  "timestamp": "2025-11-28T02:15:00Z"
}
▶️ POST /train?samples=1000
Melatih ulang model menggunakan data dummy.

Response:
json
Copy code
{
  "message": "Model retrained",
  "model_path": "ocsvm_maggot.pkl"
}
▶️ GET /status/latest
Mengambil status prediksi terbaru.

Response jika belum ada data:
json
Copy code
{
  "message": "No analysis yet"
}
🤖 Cara Kerja One-Class SVM
Model OCSVM dilatih hanya menggunakan data normal supaya model dapat mengenali pola normal tersebut.
Jika ada titik data yang berada di luar boundary model → dianggap anomaly.

📱 Integrasi ke Android (Kotlin + Retrofit)
1️⃣ Tambahkan Dependency di build.gradle
gradle
Copy code
implementation("com.squareup.retrofit2:retrofit:2.9.0")
implementation("com.squareup.retrofit2:converter-gson:2.9.0")
2️⃣ Data Class
kotlin
Copy code
data class SensorRequest(
    val temperature: Double,
    val ph: Double,
    val turbidity: Double,
    val oxygen: Double
)

data class PredictionResponse(
    val status: String,
    val score: Double,
    val timestamp: String
)
3️⃣ Retrofit Interface
kotlin
Copy code
interface ApiService {
    @POST("predict")
    suspend fun predict(@Body request: SensorRequest): PredictionResponse
}
4️⃣ Retrofit Client
kotlin
Copy code
object ApiClient {
    private const val BASE_URL = "http://192.168.1.10:8000/"

    val instance: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }
}
5️⃣ Memanggil API di Activity/ViewModel
kotlin
Copy code
CoroutineScope(Dispatchers.IO).launch {
    try {
        val request = SensorRequest(
            temperature = 31.0,
            ph = 7.2,
            turbidity = 1.8,
            oxygen = 5.0
        )

        val response = ApiClient.instance.predict(request)

        withContext(Dispatchers.Main) {
            if (response.status == "anomaly") {
                showAnomalyUI(response.score)
            } else {
                showNormalUI()
            }
        }

    } catch (e: Exception) {
        e.printStackTrace()
    }
}
6️⃣ Contoh UI Handler
kotlin
Copy code
fun showAnomalyUI(score: Double) {
    statusText.text = "⚠️ Anomaly Detected"
    statusText.setTextColor(Color.RED)
    scoreText.text = "Score: $score"
}

fun showNormalUI() {
    statusText.text = "✔ Normal"
    statusText.setTextColor(Color.GREEN)
}
🌐 Catatan untuk Developer Android
Device Android & API server harus berada pada jaringan yang sama jika testing lokal.

Ganti BASE_URL sesuai IP server.

Pastikan IoT mengirim data sesuai format JSON API.

💡 Tambahan (Opsional)
Jika ingin, bisa ditambahkan:

Logging history ke database

Dashboard web (grafik anomali)

Deployment ke cloud (Railway/Fly.io/VPS)

WebSocket untuk live data

📄 Lisensi
Proyek ini bebas digunakan untuk riset, edukasi, dan pengembangan aplikasi IoT.

❤️ Kontribusi
Pull request sangat diterima.
Silakan buka issue jika menemukan bug atau ingin menambahkan fitur baru.

yaml
Copy code

---

Kalau kamu mau, aku bisa buatkan:

✅ README versi bahasa Indonesia  
✅ Tambahkan arsitektur diagram (ASCII atau gambar)  
✅ Tambahkan tutorial deploy ke cloud  
✅ Tambahkan contoh integrasi Jetpack Compose  

Mau ditambah apa?
