function make_video(frame_dir, vid_dir, vid_name)
    if(~exist(vid_dir, 'file'))
        mkdir(vid_dir);
    end
    frame_list = dir([frame_dir '/*.png']);
    writerObj = VideoWriter([vid_dir '/' vid_name]);
    writerObj.Quality = 100;
    open(writerObj);
    num_frame = size(frame_list, 1);
    for idx_frame = 1:num_frame
        disp(['Processing frame: ' num2str(idx_frame)]);
        frame_name = frame_list(idx_frame).name;
        frame = imread([frame_dir '/' frame_name]);
        writeVideo(writerObj, frame);
    end
    close(writerObj);
end