import UploadVideo from './components/UploadVideo'
import AnalogVideo from './components/AnalogVideo'
import VideoFrame from './components/VideoFrame/VideoFrame'
import VideoInfo from './components/VideoInfo/VideoInfo'
import { useState, useRef } from 'react'
import styled from 'styled-components';

function App() {
  const videoRefAnalog = useRef(null);
  const videoRefUpload = useRef(null);
  const [videoAnalogSrc, setVideoAnalogSrc] = useState("https://www.shutterstock.com/shutterstock/videos/1093893931/preview/stock-footage-black-cat-in-water-taking-bath-black-oriental-cat-making-loud-meow-sounds-k-video-clip.webm");
  const [intervalAnalog, setintervalAnalog] = useState("0:0:0 - 0:0:1");
  const [startSecAnalog, setstartSecAnalog] = useState(0);
  const [intervalUpload, setiintervalUpload] = useState("0:0:0 - 0:0:1");
  const [startSecUpload, setstartSecUpload] = useState(0);
  const [videoNums, setVideoNums] = useState(1);
  const [curVideoNum, setCurVideoNum] = useState(1);
  const [timeNums, setTimeNums] = useState(1);
  const [curTimeNum, setCurTimeNum] = useState(1);
  const [curData, setCurData] = useState(1);
  const [showUpload, setShowUpload] = useState(true); // Добавили состояние для кнопки
  const [videoUploadSrc, setVideoUploadSrc] = useState(null);
  const titleUpload = "Загрузите видео, которое хотите проверить на нарушение авторских прав";
  const titleAnalog = "Проверенный видеофайл на нарушение авторских прав";
  const BackgroundImage = styled.div`
  background-image: url('/background.jpg'); 
  background-size: cover; // Можно использовать 'contain' или другие значения
  background-position: center; // Можно настроить позиционирование
  background-repeat: no-repeat; // Убираем повторение фона
  min-height: 100vh; // Заполняем всю высоту экрана
  /* Добавьте другие стили для вашего приложения http://127.0.0.1:8000/video/video2.mp4*/
`;

  return (
    <BackgroundImage>
      <main>
        <img src={'/rutube_logo.png'} 
             style={{ 
             width: '202px', 
             height: '33px', 
             display: 'block',
             margin: '0 auto'
           }} alt="Изображение" />
        <section style={{ display: 'flex', justifyContent: 'center' }}> 
            <VideoFrame>
              <UploadVideo videoSrc={videoUploadSrc} setShowUpload={setShowUpload} videoRef={videoRefUpload} />
            </VideoFrame>

            <VideoFrame>
              <AnalogVideo src={videoAnalogSrc} videoRef={videoRefAnalog} />
            </VideoFrame>
        </section>
        
        <div style={{ display: 'flex', justifyContent: 'center' }}> 
            <VideoInfo title={titleUpload} left={true} setVideoUploadSrc={setVideoUploadSrc} showUpload={showUpload} 
                       videoRefUpload={videoRefUpload} videoRefAnalog={videoRefAnalog} videoRef={videoRefUpload} setVideoAnalogSrc={setVideoAnalogSrc} 
                       startSec={startSecUpload} interval={intervalUpload} setintervalAnalog={setintervalAnalog}
                       setstartSecAnalog={setstartSecAnalog} setiintervalUpload={setiintervalUpload} 
                       setstartSecUpload={setstartSecUpload} curVideoNum={curVideoNum} videoNums={videoNums} 
                       curTimeNum={curTimeNum} timeNums={timeNums} setVideoNums={setVideoNums} 
                       setCurVideoNum={setCurVideoNum} setTimeNums={setTimeNums} setCurTimeNum={setCurTimeNum} 
                       curData={curData} setCurData={setCurData}/>
            <VideoInfo title={titleAnalog} left={false} setVideoUploadSrc={setVideoUploadSrc} showUpload={false} 
                       videoRefUpload={videoRefUpload} videoRefAnalog={videoRefAnalog} videoRef={videoRefAnalog} setVideoAnalogSrc={setVideoAnalogSrc} 
                       startSec={startSecAnalog} interval={intervalAnalog} setintervalAnalog={setintervalAnalog}
                       setstartSecAnalog={setstartSecAnalog} setiintervalUpload={setiintervalUpload} 
                       setstartSecUpload={setstartSecUpload} curVideoNum={curVideoNum} videoNums={videoNums}
                       curTimeNum={curTimeNum} timeNums={timeNums} setVideoNums={setVideoNums} 
                       setCurVideoNum={setCurVideoNum} setTimeNums={setTimeNums} setCurTimeNum={setCurTimeNum} 
                       curData={curData} setCurData={setCurData} />
        </div>
        
      </main>
    </BackgroundImage>
  )
}

export default App
