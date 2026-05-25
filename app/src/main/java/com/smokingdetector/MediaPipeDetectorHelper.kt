/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.smokingdetector

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
    val objectDetectorListener: DetectorListener?
) {

    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    fun setupObjectDetector() {
        val baseOptionsBuilder = BaseOptions.builder()
            .setModelAssetPath("cigarette-detector.tflite")

        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionsBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
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

    @Synchronized
    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        // MediaPipe handles rotation internally via its ImageProcessor/BitmapImageBuilder
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

    @Synchronized
    fun clear() {
        objectDetector?.close()
        objectDetector = null
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
            results: MutableList<Detection>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
    }
}
