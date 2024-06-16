import React, { useRef } from 'react';

export default function UploadVideo({ videoSrc, setShowUpload, videoRef }) {

  const handleShowUpload = () => {
    setShowUpload(true); // Показываем кнопку снова
  };

  return (
    <div>
      {videoSrc && (
        <div>
          <video 
            key={videoSrc} 
            ref={videoRef} 
            width="640" 
            height="360" 
            controls 
            onEnded={handleShowUpload} // Показываем кнопку после окончания видео
          >
            <source src={videoSrc} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      )}  
    </div>
  );
}
