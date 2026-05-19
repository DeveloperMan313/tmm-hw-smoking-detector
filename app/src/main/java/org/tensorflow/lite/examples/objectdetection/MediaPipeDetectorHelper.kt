package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import org.tensorflow.lite.task.vision.detector.Detection

class MediaPipeDetectorHelper(
    var threshold: Float = 0.5f,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    val context: Context,
    val objectDetectorListener: ObjectDetectorHelper.DetectorListener?
) {

    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    fun setupObjectDetector() {
        val baseOptionsBuilder = BaseOptions.builder()
            .setModelAssetPath("cigarette-detector.tflite")

        when (currentDelegate) {
            ObjectDetectorHelper.DELEGATE_CPU -> {
                baseOptionsBuilder.setDelegate(Delegate.CPU)
            }
            ObjectDetectorHelper.DELEGATE_GPU -> {
                baseOptionsBuilder.setDelegate(Delegate.GPU)
            }
        }

        val optionsBuilder = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptionsBuilder.build())
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
            .setRunningMode(RunningMode.IMAGE)

        try {
            objectDetector = ObjectDetector.createFromOptions(context, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                "MediaPipe Object detector failed to initialize. See error logs for details"
            )
            Log.e("MediaPipe", "TFLite failed to load model with error: " + e.message)
        } catch (e: RuntimeException) {
            objectDetectorListener?.onError(
                "MediaPipe Object detector failed to initialize. See error logs for details"
            )
            Log.e("MediaPipe", "Runtime error: " + e.message)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        // MediaPipe handles rotation internally via its ImageProcessor/BitmapImageBuilder
        // But for RunningMode.IMAGE we just pass the bitmap.
        val mpImage = BitmapImageBuilder(image).build()

        val results = objectDetector?.detect(mpImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        objectDetectorListener?.onResults(
            results?.toTfliteDetections(image.width, image.height),
            inferenceTime,
            image.height,
            image.width
        )
    }

    // Convert MediaPipe Detection results to TFLite Task Library Detection objects 
    // to reuse the existing OverlayView and listener.
    private fun ObjectDetectorResult.toTfliteDetections(
        imageWidth: Int,
        imageHeight: Int
    ): MutableList<Detection> {
        val tfliteDetections = mutableListOf<Detection>()
        for (mpDetection in this.detections()) {
            val categories = mutableListOf<org.tensorflow.lite.support.label.Category>()
            for (mpCategory in mpDetection.categories()) {
                categories.add(
                    org.tensorflow.lite.support.label.Category.create(
                        mpCategory.categoryName(),
                        mpCategory.displayName(),
                        mpCategory.score(),
                        mpCategory.index()
                    )
                )
            }

            // magic constants because i don't know what i'm doing
            val mpBox = mpDetection.boundingBox()
            val flippedBox = RectF(
                (imageWidth.toFloat() * 0.75f - mpBox.bottom) * 0.75f,
                mpBox.left * 0.75f,
                (imageWidth.toFloat() * 0.75f - mpBox.top) * 0.75f,
                mpBox.right * 0.75f
            )

            tfliteDetections.add(
                Detection.create(flippedBox, categories)
            )
        }
        return tfliteDetections
    }

    fun clear() {
        objectDetector?.close()
        objectDetector = null
    }
}
