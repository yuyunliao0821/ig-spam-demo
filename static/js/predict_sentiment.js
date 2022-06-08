window.onload = function(){
  function predictSpam() {
  
    var inputText = document.getElementById("textbox").value;
    var likes = document.getElementById('likes').value;
    var comments = document.getElementById('comments').value;
    var predictionContainer = document.getElementById("prediction");
    var predictionText = document.getElementById("prediction-text");
  
    predictionContainer.style.display = 'flex';
    predictionText.style.display= 'None'
  
    var serverData=[{
      'text':inputText,
      'likes':likes,
      'comments':comments,
    }]
  
    $.post({
      url:"/predict", 
      data:JSON.stringify(serverData), 
      contentType:'application/json'}).done(function (data){
        predictionText.style.display = 'flex';
        predictionText.textContent = data;
      }
    );
  
    //  $.get("/predict?"+"text="+inputText+"&"+"mlmodel="+selectedModel).done(function (data) {
    //    predictionText.innerHTML = data;
    //  })
  }
  
  var button = document.getElementById('button')
  button.addEventListener('click', predictSpam)
}
