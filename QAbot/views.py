from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse
from .static.decode_seq2seq import answer
def ready_view(request):
    return render(request,'qabot.html')

def reply(request):
    if request.method == "POST":
        sentence = request.POST['sentence']
        bot_answer = answer(sentence)
        return HttpResponse(bot_answer)
    return HttpResponse("FAIL!!!")
# Create your views here.
