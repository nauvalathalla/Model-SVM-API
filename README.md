# 🚀 OCSVM Anomaly Detection API -- Integration Guide (Android)

Repositori ini berisi layanan backend untuk melakukan **anomaly
detection** menggunakan **One-Class SVM (OCSVM)**. Backend ini
menyediakan **REST API** agar aplikasi Android dapat mengambil hasil
prediksi dan menampilkannya di UI.

------------------------------------------------------------------------

## 📌 Fitur Utama

-   Machine Learning dengan **One-Class SVM (OCSVM)**
-   API prediksi menggunakan **FastAPI**
-   Penyimpanan hasil model ke **PostgreSQL**
-   Scheduler untuk pengecekan berkala
-   Siap diintegrasikan ke aplikasi Android / Kotlin

------------------------------------------------------------------------

# 🏗️ Arsitektur Singkat

    Android App → FastAPI Server → OCSVM Model → PostgreSQL

Android hanya perlu **memanggil endpoint** dan menampilkan response.

------------------------------------------------------------------------

# ⚙️ 1. Cara Menjalankan Backend

## Install Dependency

``` bash
pip install -r requirements.txt
```

## Jalankan Server

``` bash
uvicorn main:app --reload
```

Server akan berjalan di:

    http://localhost:8000

------------------------------------------------------------------------

# 📡 2. Endpoint API

## 🔎 **1. Prediksi Data**

**POST** `/predict`

**Body (JSON):**

``` json
{
  "value": 12.5
}
```

**Response:**

``` json
{
  "status": "normal",
  "score": 0.23
}
```

------------------------------------------------------------------------

## 📥 **2. Ambil Hasil Prediksi Terakhir**

**GET** `/latest`

**Response:**

``` json
{
  "timestamp": "2025-11-28T13:20:00",
  "value": 12.5,
  "status": "anomaly",
  "score": -0.4
}
```

------------------------------------------------------------------------

# 📱 3. Cara Integrasi ke Android (Kotlin)

## Tambahkan Dependency

``` gradle
implementation("com.squareup.retrofit2:retrofit:2.9.0")
implementation("com.squareup.retrofit2:converter-gson:2.9.0")
```

------------------------------------------------------------------------

# 🔧 API Client (Retrofit)

``` kotlin
interface ApiService {
    @GET("latest")
    suspend fun getLatestStatus(): LatestResponse

    @POST("predict")
    suspend fun predict(@Body body: PredictBody): PredictResponse
}

data class LatestResponse(
    val timestamp: String,
    val value: Double,
    val status: String,
    val score: Double
)

data class PredictBody(val value: Double)

data class PredictResponse(
    val status: String,
    val score: Double
)
```

------------------------------------------------------------------------

# 📲 Cara Menampilkan Data di UI

``` kotlin
viewModelScope.launch {
    try {
        val result = api.getLatestStatus()
        _state.value = "Status: ${result.status}\nScore: ${result.score}"
    } catch (e: Exception) {
        _state.value = "Error: ${e.message}"
    }
}
```

------------------------------------------------------------------------

# 🖼️ Contoh Tampilan di Jetpack Compose

``` kotlin
@Composable
fun StatusCard(state: String) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        elevation = CardDefaults.cardElevation(8.dp)
    ) {
        Text(
            text = state,
            modifier = Modifier.padding(16.dp),
            style = MaterialTheme.typography.bodyLarge
        )
    }
}
```

------------------------------------------------------------------------

# 📦 Struktur Repository

    ├── main.py
    ├── ocsvm.py
    ├── scheduler.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# 🤝 Kontribusi

Pull Request sangat diterima.

------------------------------------------------------------------------

# 📬 Kontak

Nauval Yusriya Athalla
