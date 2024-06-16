import React, { useRef, useState } from 'react';

export default function VideoPlayer({ src, videoRef }) {

return (
  <div>
    <video
      ref={videoRef}
      src={src}
      width="640"
      height="360"
      controls
    >
      Your browser does not support the video tag.
    </video>
  </div>
);
}
