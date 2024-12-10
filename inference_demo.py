import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from siamrpn import TrackerSiamRPN
import tempfile
import os
from PIL import Image
import time
from tqdm import tqdm
import ffmpeg

def track_with_progress(tracker, frames, box, st_progress):
    """Tracking with streamlit progress bar"""
    frame_num = len(frames)
    boxes = np.zeros((frame_num, 4))
    boxes[0] = box
    times = np.zeros(frame_num)
    
    # 计算总帧数用于进度显示
    total_frames = len(frames)
    
    for f, img in enumerate(frames):
        # 更新进度条
        progress = int((f + 1) * 100 / total_frames)
        st_progress.progress(progress, f"处理第 {f+1}/{total_frames} 帧...")
        
        start_time = time.time()
        if f == 0:
            tracker.init(img, box)
        else:
            boxes[f, :] = tracker.update(img)
        times[f] = time.time() - start_time
    
    return boxes, times

def main():
    st.title("单目标跟踪演示")
    
    # 模型加载
    @st.cache_resource
    def load_model():
        net_path = 'pretrained/siamrpn/model.pth'
        return TrackerSiamRPN(net_path=net_path)
    
    tracker = load_model()

    # 会话状态初始化
    if 'frames' not in st.session_state:
        st.session_state.frames = None
    if 'roi_selected' not in st.session_state:
        st.session_state.roi_selected = False
    if 'tracking_started' not in st.session_state:
        st.session_state.tracking_started = False

    # 上传视频文件
    video_file = st.file_uploader("上传视频文件", type=['mp4', 'avi'])
    
    if video_file is not None and st.session_state.frames is None:
        with st.spinner("正在读取视频..."):
            try:
                # 保存上传的视频到临时文件
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.getvalue())
                tfile.flush()
                tfile.close()

                # 使用 ffmpeg 读取视频信息
                probe = ffmpeg.probe(tfile.name)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                width = int(video_info['width'])
                height = int(video_info['height'])
                total_frames = int(video_info['nb_frames'])

                st.write(f"视频信息: {width}x{height}, {total_frames}帧")

                # 使用 ffmpeg 读取视频帧
                out, _ = (
                    ffmpeg
                    .input(tfile.name)
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True)
                )
                
                # 直接使用 RGB 格式的帧
                frames = []
                progress_bar = st.progress(0)
                video_frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
                
                for i, frame in enumerate(video_frames):
                    # 直接使用 RGB 格式，不需要转换
                    frames.append(frame)
                    progress_bar.progress((i + 1) / len(video_frames))

                os.unlink(tfile.name)
                
                if len(frames) > 0:
                    st.session_state.frames = frames
                    st.success(f"成功读取 {len(frames)} 帧")
                    
                    # 显示第一帧
                    if not st.session_state.roi_selected:
                        st.image(frames[0], caption="第一帧", use_container_width=True)
                        st.info("请在图像上框选目标区域")
                else:
                    st.error("没有读取到任何帧")
                    
            except Exception as e:
                st.error(f"处理视频时发生错误: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.frames is not None:
        st.write("在图像上拖动鼠标来选择跟踪目标区域")
        
        # 获取第一帧的尺寸
        first_frame = st.session_state.frames[0]
        h, w = first_frame.shape[:2]
        
        # 计算适合显示的尺寸
        max_display_width = 800  # 最大显示宽度
        display_scale = min(max_display_width / w, 1.0)  # 缩放比例
        display_width = int(w * display_scale)
        display_height = int(h * display_scale)
        
        # 调整第一帧大小用于显示
        display_frame = cv2.resize(first_frame, (display_width, display_height))
        
        # 创建画布让用户选择ROI
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#00ff00",
            background_image=Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)),
            drawing_mode="rect",
            key="canvas",
            width=display_width,
            height=display_height,
        )
        
        # 获取用户绘制的矩形并转换回原始尺寸
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            # 获取最后绘制的矩形
            rect = canvas_result.json_data["objects"][-1]
            # 转换坐标到原始尺寸
            x = int(rect["left"] / display_scale)
            y = int(rect["top"] / display_scale)
            w = int(rect["width"] / display_scale)
            h = int(rect["height"] / display_scale)
            
            if st.button("确认选择并开始跟踪"):
                # 创建进度条
                progress_bar = st.progress(0)
                
                try:
                    # 执行跟踪
                    boxes, times = track_with_progress(
                        tracker, 
                        st.session_state.frames,
                        box=[x, y, w, h],
                        st_progress=progress_bar
                    )
                    
                    # 保存跟踪结果到会话状态
                    st.session_state.boxes = boxes
                    st.session_state.times = times
                    st.session_state.tracking_started = True
                    
                    # 显示完成信息
                    st.success(f"跟踪完成！总用时: {np.sum(times):.2f} 秒")
                    
                except Exception as e:
                    st.error(f"跟踪过程中出现错误: {str(e)}")
                
                # 重新加载页面以显示结果
                # st.experimental_rerun()
        
        # 显示跟踪结果
        if st.session_state.tracking_started:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 创建帧选择滑块
                frame_idx = st.slider("选择帧", 0, len(st.session_state.frames)-1, 0)
                
                # 显示选定帧的跟踪结果
                frame_with_result = st.session_state.frames[frame_idx].copy()
                box = st.session_state.boxes[frame_idx]
                cv2.rectangle(frame_with_result, 
                            (int(box[0]), int(box[1])), 
                            (int(box[0] + box[2]), int(box[1] + box[3])), 
                            (0, 255, 0), 2)
                
                # 调整显示大小
                frame_with_result = cv2.resize(frame_with_result, (display_width, display_height))
                st.image(frame_with_result, channels="BGR", use_container_width=True)
            
            with col2:
                # 显示跟踪信息（显示原始尺寸的坐标）
                st.write("跟踪信息")
                st.write(f"帧: {frame_idx + 1}/{len(st.session_state.frames)}")
                st.write(f"处理时间: {st.session_state.times[frame_idx]:.3f} 秒")
                st.write("边界坐标 (原始尺寸):")
                st.write(f"x: {box[0]:.1f}")
                st.write(f"y: {box[1]:.1f}")
                st.write(f"宽: {box[2]:.1f}")
                st.write(f"高: {box[3]:.1f}")

            # 保存处理后的视频
            if st.session_state.tracking_started and len(st.session_state.frames) > 0:
                if st.button("保存跟踪结果视频"):
                    # 使用临时目录来保存视频
                    temp_dir = tempfile.gettempdir()
                    output_path = os.path.join(temp_dir, "tracking_result.mp4")
                    
                    with st.spinner("正在保存视频..."):
                        try:
                            height, width = st.session_state.frames[0].shape[:2]
                            
                            # 创建带有跟踪框的帧
                            tracking_frames = []
                            for frame, bbox in zip(st.session_state.frames, st.session_state.boxes):
                                frame_with_box = frame.copy()
                                if bbox is not None:
                                    # 绘制跟踪框
                                    x, y, w, h = [int(v) for v in bbox]
                                    cv2.rectangle(frame_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                tracking_frames.append(frame_with_box)
                            
                            # 尝试不同的编码器
                            try:
                                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            except:
                                try:
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                except:
                                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            
                            # 创建视频写入器
                            out = cv2.VideoWriter(
                                output_path,
                                fourcc,
                                30.0,
                                (width, height),
                                isColor=True
                            )
                            
                            if not out.isOpened():
                                # 使用 ffmpeg 写入
                                frames_array = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in tracking_frames])
                                
                                process = (
                                    ffmpeg
                                    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=30)
                                    .output(output_path, pix_fmt='yuv420p', vcodec='libx264')
                                    .overwrite_output()
                                    .run_async(pipe_stdin=True)
                                )
                                
                                for frame in frames_array:
                                    process.stdin.write(frame.tobytes())
                                
                                process.stdin.close()
                                process.wait()
                                
                            else:
                                # 使用 OpenCV 写入
                                for frame in tracking_frames:
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    out.write(frame_bgr)
                                out.release()
                            
                            # 检查文件是否成功创建
                            if not os.path.exists(output_path):
                                raise Exception("视频文件未能成功创建")
                            
                            # 提供下载链接
                            with open(output_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            st.download_button(
                                label="下载跟踪结果视频",
                                data=video_bytes,
                                file_name="tracking_result.mp4",
                                mime="video/mp4"
                            )
                            
                            st.success(f"视频已保存并准备下载")
                            
                        except Exception as e:
                            st.error(f"保存视频时发生错误: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                        finally:
                            try:
                                if os.path.exists(output_path):
                                    os.remove(output_path)
                            except Exception as e:
                                st.warning(f"清理临时文件时发生错误: {str(e)}")

                            # 添加调试信息
                            st.write(f"临时目录路径: {temp_dir}")
                            st.write(f"输出文件路径: {output_path}")
                            st.write(f"当前工作目录: {os.getcwd()}")

if __name__ == "__main__":
    main()
