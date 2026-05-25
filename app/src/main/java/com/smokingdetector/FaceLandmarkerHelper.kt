/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

class FaceLandmarkerHelper(
    var minFaceDetectionConfidence: Float = 0.5f,
    var minFacePresenceConfidence: Float = 0.5f,
    var minTrackingConfidence: Float = 0.5f,
    var maxNumFaces: Int = 1,
    var currentDelegate: Int = 0,
    val context: Context,
    val faceLandmarkerListener: LandmarkerListener?
) {

    private var faceLandmarker: FaceLandmarker? = null

    init {
        setupFaceLandmarker()
    }

    fun clear() {
        faceLandmarker?.close()
        faceLandmarker = null
    }

    fun setupFaceLandmarker() {
        val baseOptionsBuilder = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")

        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionsBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionsBuilder.setDelegate(Delegate.GPU)
            }
        }

        try {
            val optionsBuilder =
                FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(baseOptionsBuilder.build())
                    .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
                    .setMinFacePresenceConfidence(minFacePresenceConfidence)
                    .setMinTrackingConfidence(minTrackingConfidence)
                    .setNumFaces(maxNumFaces)
                    .setRunningMode(RunningMode.IMAGE)

            faceLandmarker =
                FaceLandmarker.createFromOptions(context, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            faceLandmarkerListener?.onError(
                "Face landmarker failed to initialize. See error logs for details"
            )
            Log.e(
                TAG, "MediaPipe failed to load model with error: " + e.message
            )
        } catch (e: RuntimeException) {
            faceLandmarkerListener?.onError(
                "Face landmarker failed to initialize. See error logs for details"
            )
            Log.e(
                TAG, "MediaPipe failed to load model with error: " + e.message
            )
        }
    }

    fun detect(image: Bitmap) {
        if (faceLandmarker == null) {
            setupFaceLandmarker()
        }

        var inferenceTime = SystemClock.uptimeMillis()

        val mpImage = BitmapImageBuilder(image).build()

        val result = faceLandmarker?.detect(mpImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (result != null) {
            faceLandmarkerListener?.onResults(
                result,
                inferenceTime,
                image.height,
                image.width
            )
        }
    }

    interface LandmarkerListener {
        fun onError(error: String)
        fun onResults(
            resultBundle: FaceLandmarkerResult,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        private const val TAG = "FaceLandmarkerHelper"
    }
}
