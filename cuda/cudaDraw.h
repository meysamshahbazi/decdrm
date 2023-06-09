/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __CUDA_DRAW_H__
#define __CUDA_DRAW_H__


#include "cudaUtility.h"
#include "imageFormat.h"


/**
 * cudaDrawCircle
 * @ingroup drawing
 */
cudaError_t cudaDrawCircle( void* input, void* output, size_t width, size_t height, imageFormat format, 
					   int cx, int cy, float radius, const float4& color );

/**
 * @brief my custom function for drawwing on Y channle of YUV image 
 * 
 * @param input 
 * @param output 
 * @param width 
 * @param height 
 * @param format 
 * @param cx 
 * @param cy 
 * @param radius 
 * @param color 
 * @return * cudaError_t 
 */
cudaError_t cudaDrawCircleOnY( void* input, void* output, size_t width, size_t height, imageFormat format,
						int cx, int cy, float radius, const float4& color );

/**
 * @brief my custom function that convert rgb color to yuv and draw circle with givven collor to yuv image 
 * 
 * @param input_y 
 * @param input_u 
 * @param input_v 
 * @param width 
 * @param height 
 * @param format 
 * @param cx 
 * @param cy 
 * @param radius 
 * @param color 
 * @return cudaError_t 
 */
cudaError_t cudaDrawCircleOnYUV420( void* input_y, void* input_u,void* input_v, size_t width, size_t height, imageFormat format, 
					int cx, int cy, float radius, const float4& color );

cudaError_t cudaDrawCircleOnYUYU( void* input, size_t width, size_t height,
					int cx, int cy, float radius, const float4& color );

/**
 * cudaDrawCircle
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawCircle( T* input, T* output, size_t width, size_t height, 
				 	   int cx, int cy, float radius, const float4& color )	
{ 
	return cudaDrawCircle(input, output, width, height, imageFormatFromType<T>(), cx, cy, radius, color); 
}	

cudaError_t cudaDeinterlace(void* input_cur, void* output, size_t width, size_t height);

cudaError_t cudaAlongSideYUYV( void* input_0, void* input_1, void* output, size_t width, size_t height);

/**
 * cudaDrawCircle (in-place)
 * @ingroup drawing
 */
inline cudaError_t cudaDrawCircle( void* image, size_t width, size_t height, imageFormat format, 
							int cx, int cy, float radius, const float4& color )
{
	return cudaDrawCircle(image, image, width, height, format, cx, cy, radius, color);
}

cudaError_t cudaDrawCircleBoarder(void* input, void* output, size_t width, size_t height, imageFormat format, int cx, int cy, float radius_1,float radius_2, const float4& color );

cudaError_t cudaDrawPayloadYUYV(void* input, int c_x, int c_y, float radius, int boarder, bool has_cross,
				const float4& color_boarder,const float4& color_background);

/**
 * cudaDrawCircle (in-place)
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawCircle( T* image, size_t width, size_t height, 
				 	   int cx, int cy, float radius, const float4& color )	
{ 
	return cudaDrawCircle(image, width, height, imageFormatFromType<T>(), cx, cy, radius, color); 
}


/**
 * cudaDrawLine
 * @ingroup drawing
 */
cudaError_t cudaDrawLine( void* input, void* output, size_t width, size_t height, imageFormat format, 
					 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 );
	
/**
 * cudaDrawLine
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawLine( T* input, T* output, size_t width, size_t height, 
				 	 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )	
{ 
	return cudaDrawLine(input, output, width, height, imageFormatFromType<T>(), x1, y1, x2, y2, color, line_width); 
}

/**
 * cudaDrawLine (in-place)
 * @ingroup drawing
 */
inline cudaError_t cudaDrawLine( void* image, size_t width, size_t height, imageFormat format, 
						   int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )
{
	return cudaDrawLine(image, image, width, height, format, x1, y1, x2, y2, color, line_width);
}					
	
/**
 * cudaDrawLine (in-place)
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawLine( T* image, size_t width, size_t height, 
				 	 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )	
{ 
	return cudaDrawLine(image, width, height, imageFormatFromType<T>(), x1, y1, x2, y2, color, line_width); 
}	


/**
 * cudaDrawRect
 * @ingroup drawing
 */
cudaError_t cudaDrawRect( void* input, void* output, size_t width, size_t height, imageFormat format, 
					 int left, int top, int right, int bottom, const float4& color, 
					 const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f );

/**
 * cudaDrawRect
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawRect( T* input, T* output, size_t width, size_t height, 
				 	 int left, int top, int right, int bottom, const float4& color,
					 const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f )	
{ 
	return cudaDrawRect(input, output, width, height, imageFormatFromType<T>(), left, top, right, bottom, color, line_color, line_width); 
}

/**
 * cudaDrawRect (in-place)
 * @ingroup drawing
 */
inline cudaError_t cudaDrawRect( void* image, size_t width, size_t height, imageFormat format, 
						   int left, int top, int right, int bottom, const float4& color,
						   const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f )
{
	return cudaDrawRect(image, image, width, height, format, left, top, right, bottom, color, line_color, line_width);
}

/**
 * cudaDrawRect
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawRect( T* image, size_t width, size_t height, 
				 	 int left, int top, int right, int bottom, const float4& color,
					 const float4& line_color=make_float4(0,0,0,0), float line_width=1.0f )	
{ 
	return cudaDrawRect(image, image, width, height, imageFormatFromType<T>(), left, top, right, bottom, color, line_color, line_width); 
}

cudaError_t cudaPutLogo( void* input, void* logo, 
						size_t input_width, size_t input_height,
						size_t logo_width, size_t logo_height, 
						size_t logo_pitch, int cx, int cy);

cudaError_t cudaDrawPayload(void* input, int c_x, int c_y, float radius, int boarder, bool has_cross,
				const float4& color_boarder,const float4& color_background);

				
#endif
