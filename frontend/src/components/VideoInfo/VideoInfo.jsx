import './VideoInfo.css'
import { useUploadVideo } from '/src/hooks/useUploadVideo'
import React, { useRef, useState } from 'react';

export default function VideoInfo({ title, left, setVideoUploadSrc, showUpload, videoRefUpload, videoRefAnalog, videoRef, setVideoAnalogSrc, startSec, 
  interval, setintervalAnalog, setstartSecAnalog, setiintervalUpload, setstartSecUpload,  curVideoNum, videoNums, curTimeNum, timeNums, setVideoNums, 
  setCurVideoNum, setTimeNums, setCurTimeNum, curData, setCurData }) {

  const {uploadVideo} = useUploadVideo({ setVideoUploadSrc, setVideoAnalogSrc, setintervalAnalog, setstartSecAnalog, setiintervalUpload, setstartSecUpload, 
    setVideoNums, setCurVideoNum, setTimeNums, setCurTimeNum, setCurData, videoRefUpload, videoRefAnalog })

  const handleFileChange = (event) => {
    uploadVideo(event.target.files[0])
  };

  const getButtonNumsText = () => `Номер видео: ${curVideoNum} / ${videoNums}`;

  const getButtonTimeText = () => `Номер интервала: ${curTimeNum} / ${timeNums}`;

  const getButtonTime = () => `Интервал: ${interval}`;

  const handleNumTimeChange = () => {
    var j = 0;
    if (curTimeNum < timeNums) {
      setCurTimeNum(curTimeNum + 1);
      j = curTimeNum;
    } else {
      setCurTimeNum(1);
    }
    setstartSecAnalog(curData.analog_info[curVideoNum - 1].time_intervals[j].start_sec)
    setintervalAnalog(curData.analog_info[curVideoNum - 1].time_intervals[j].t_start)
    setstartSecUpload(curData.upload_info[curVideoNum - 1][j].start_sec)
    setiintervalUpload(curData.upload_info[curVideoNum - 1][j].t_start)
  };

  const handleVideoChange = () => {
    var i = 0;
    var num = 0;
    if (curVideoNum < videoNums) {
      setCurVideoNum(curVideoNum + 1);
      i = curVideoNum;
    } else {
      setCurVideoNum(1);
    }
    setVideoAnalogSrc(curData.analog_info[i].filename)
    setTimeNums(curData.analog_info[i].time_intervals.length)
    setCurTimeNum(1)
    setstartSecAnalog(curData.analog_info[i].time_intervals[0].start_sec)
    setintervalAnalog(curData.analog_info[i].time_intervals[0].t_start)
    setstartSecUpload(curData.upload_info[i][0].start_sec)
    setiintervalUpload(curData.upload_info[i][0].t_start)
  };

  const handleTimeChange = (new_time) => {
    videoRef.current.currentTime = new_time;
  };

  return (
    <div style={{ display: "block"}}>
      <p className={left ? "text-video-style-left" : "text-video-style-right"}>
        {title}
      </p>
      {showUpload && (
        <input className="upload-button-video-style-left" type="file" accept="video/*" onChange={handleFileChange} />
      )}
      {!left && (
        <button onClick={handleVideoChange} className={"button-video-num-style-right"}>
          {getButtonNumsText()}
        </button>)}
      {!left && (
        <button onClick={handleNumTimeChange} className={"time-num-button-video-style-right"}>
          {getButtonTimeText()}
        </button>
      )}
      <button onClick={handleTimeChange.bind(this, startSec)} className={left ? "time-button-video-style-left" : "time-button-video-style-right"}>
        {getButtonTime()}
      </button>
    </div>
  );   
}
