/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.LinkedList
import kotlin.math.max
import org.tensorflow.lite.task.vision.detector.Detection

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<Detection> = LinkedList<Detection>()
    private var faceLandmarkerResult: FaceLandmarkerResult? = null
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var facePaint = Paint()

    private var scaleFactor: Float = 1f

    private var bounds = Rect()
    private var outputImageHeight: Int = 1
    private var outputImageWidth: Int = 1

    init {
        initPaints()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        facePaint.reset()
        faceLandmarkerResult = null
        results = LinkedList<Detection>()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE

        facePaint.color = Color.GREEN
        facePaint.strokeWidth = 2f
        facePaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        for (result in results) {
            val boundingBox = result.boundingBox

            val top = boundingBox.top * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor
            val left = boundingBox.left * scaleFactor
            val right = boundingBox.right * scaleFactor

            // Draw bounding box around detected objects
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Create text to display alongside detected objects
            val drawableText =
                result.categories[0].label + " " +
                        String.format("%.2f", result.categories[0].score)

            // Draw rect behind display text
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + Companion.BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + Companion.BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Draw text for detected object
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }

        faceLandmarkerResult?.let { result ->
            for (faceLandmarks in result.faceLandmarks()) {
                for (connector in FaceLandmarker.FACE_LANDMARKS_CONNECTORS) {
                    val start = faceLandmarks[connector.start()]
                    val end = faceLandmarks[connector.end()]

                    // Convert normalized landmarks to pixels
                    val startPixelX = start.x() * outputImageWidth
                    val startPixelY = start.y() * outputImageHeight
                    val endPixelX = end.x() * outputImageWidth
                    val endPixelY = end.y() * outputImageHeight

                    // Apply the same "magick" transformation as in MediaPipeDetectorHelper
                    // newX = (imageWidth * 0.75 - pixelY) * 0.75
                    // newY = pixelX * 0.75
                    val transformedStartX = (outputImageWidth * 0.75f - startPixelY) * 0.75f
                    val transformedStartY = startPixelX * 0.75f
                    val transformedEndX = (outputImageWidth * 0.75f - endPixelY) * 0.75f
                    val transformedEndY = endPixelX * 0.75f

                    canvas.drawLine(
                        transformedStartX * scaleFactor,
                        transformedStartY * scaleFactor,
                        transformedEndX * scaleFactor,
                        transformedEndY * scaleFactor,
                        facePaint
                    )
                }
            }
        }
    }

    fun setResults(
      detectionResults: MutableList<Detection>,
      imageHeight: Int,
      imageWidth: Int,
    ) {
        results = detectionResults
        outputImageHeight = imageHeight
        outputImageWidth = imageWidth

        // PreviewView is in FILL_START mode. So we need to scale up the bounding box to match with
        // the size that the captured images will be displayed.
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
    }

    fun setFaceResults(
        result: FaceLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
    ) {
        faceLandmarkerResult = result
        outputImageHeight = imageHeight
        outputImageWidth = imageWidth
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
