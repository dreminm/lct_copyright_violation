export const useUploadVideo = ({setVideoUploadSrc, setVideoAnalogSrc, setintervalAnalog, setstartSecAnalog, setiintervalUpload, setstartSecUpload,
  setVideoNums, setCurVideoNum, setTimeNums, setCurTimeNum, setCurData, videoRefUpload, videoRefAnalog }) => {
    const uploadVideo = (file) => {
        if (file) {
            const reader = new FileReader();
      
            reader.onload = (e) => {
              setVideoUploadSrc(e.target.result);
            };
      
            reader.readAsDataURL(file);
      
            // Отправка данных на сервер
            const formData = new FormData();
            formData.append('video', file); // 'video' - имя поля на сервере 
      
            fetch('http://127.0.0.1:8080/upload', { // Замените '/upload' на ваш URL 
              method: 'POST',
              body: formData 
            })
            .then(response => {
              if (!response.ok) {
                throw new Error('Ошибка загрузки видео');
              }
              
              return response.json(); //  Если сервер возвращает JSON
            })
            .then(data => {
              setCurData(data)
              setVideoAnalogSrc(data.analog_info[0].filename)
              setstartSecAnalog(data.analog_info[0].time_intervals[0].start_sec)
              setintervalAnalog(data.analog_info[0].time_intervals[0].t_start)
              setstartSecUpload(data.upload_info[0][0].start_sec)
              setiintervalUpload(data.upload_info[0][0].t_start)
              setVideoNums(data.analog_info.length)
              setCurVideoNum(1)
              setTimeNums(data.analog_info[0].time_intervals.length)
              setCurTimeNum(1)
              videoRefUpload.current.currentTime = data.upload_info[0][0].start_sec
              videoRefAnalog.current.currentTime = data.analog_info[0].time_intervals[0].start_sec
            })
            .catch(error => {
              console.error('Ошибка:', error);
            });
          }
    }
    return {
        uploadVideo
    }
}