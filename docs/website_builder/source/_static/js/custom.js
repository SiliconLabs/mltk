
window.SURVEY_URL = 'https://www.surveymonkey.com/r/JDGSDJC';
window.SHOW_SURVEY_AFTER_SECONDS = 3*60; // Show the survey after 3min of activity

window.dataLayer = window.dataLayer || [];
window.loadingSurvey = false;
window.showingSurvey = false;
window.tookSurvey = localStorage.surveyUrl === window.SURVEY_URL


// Open external links in a new tab
$(document).ready(function () {
    addScrollToTopButton();
    checkIfAcceptedCookies();
    $('a[href^="http://"], a[href^="https://"]').not('a[class*=internal]').attr('target', '_blank');
});


function gtag() {
    dataLayer.push(arguments);
}

function initialiseGoogleAnalytics() {
    console.log('Loading google analytics');
    gtag('js', new Date());
    gtag('config', gTrackingId, {'anonymize_ip': true});
}

function checkIfAcceptedCookies() {
    if (!localStorage.acceptedCookies) {
        $('.privacy-banner').show();
        $('.privacy-banner-accept').click(function() {
            $('.privacy-banner').hide()
            localStorage.acceptedCookies = 'true';
            onAcceptedCookies();
        });
        
    } else {
        onAcceptedCookies();
    }
}

function onAcceptedCookies() {
    initialiseGoogleAnalytics();
    initializeSurvey();
}


function initializeSurvey() {
    $('#survey-link').on('click', function() {
        window.tookSurvey = false;
        window.loadingSurvey = false;
        window.showingSurvey = false;
        showSurvey();
    });

    if(!window.tookSurvey) {
        // If the survey was previously taken, but a new survey URL is available
        // then reset the activity counter
        if(localStorage.surveyUrl && localStorage.surveyUrl !== window.SURVEY_URL) {
            localStorage.activeSeconds = 0;
        } 
        // Otherwise, if no counter has been previously set
        // then set one now
        else if(!localStorage.activeSeconds) {
            localStorage.activeSeconds = 0;
        }

        localStorage.lastActivityTimestamp = Date.now() / 1000;

        // Track mouse movement and scrolling
        $(window).scroll(updateSurveyActivity);
        $(document).on('mousemove', updateSurveyActivity);
    }
}


function showSurvey() {
    if(localStorage.acceptedCookies && !window.tookSurvey && !window.loadingSurvey) {
        window.loadingSurvey = true;

        $('#iframe-survey').on('load', function() {
            if(window.showingSurvey) {
                closeSurvey();
            } else {
                window.showingSurvey = true;
                $('#dlg-survey').css('display', 'block');
            }
        });
        $('#iframe-survey').attr('src', window.SURVEY_URL);
        $("#dlg-survey-close").on("click", closeSurvey);
    }
}

function closeSurvey() {
    console.info('Took survey');
    window.tookSurvey = true;
    localStorage.surveyUrl = window.SURVEY_URL;
    $('#dlg-survey').css('display', 'none');
    $('#iframe-survey').off('load');
    $('#iframe-survey').attr('src', '');
}

function updateSurveyActivity() {
    let now = Date.now() / 1000;
    let elapsed = now - parseFloat(localStorage.lastActivityTimestamp);
    
    if(!elapsed || elapsed < 1) { 
        return
    }
    localStorage.lastActivityTimestamp = now;

    if(elapsed > 5*60) { // If more than 5min elapsed, then assume the user walked away so ignore this activity
        return;
    }

    let totalSeconds = parseFloat(localStorage.activeSeconds) + elapsed;
    localStorage.activeSeconds = totalSeconds;
    if(totalSeconds >= window.SHOW_SURVEY_AFTER_SECONDS) {
        showSurvey();
    }
}


function addScrollToTopButton() {
    $(window).scroll(function() {
        var footertotop = ($('footer').position().top);
        var scrolltop = $(document).scrollTop() + window.innerHeight;
        var difference = scrolltop-footertotop - 30;

        if (scrolltop > footertotop) {
            $('.go-top').css({'bottom' : difference});
        }else{
            $('.go-top').css({'bottom' : 10});
        };   

        if ($(this).scrollTop() > 200) {
            $('.go-top').fadeIn(200);
        } else {
            $('.go-top').fadeOut(200);
        }
    });

    $('.go-top').click(function(event) {
        event.preventDefault();
        $('html, body').animate({scrollTop: 0}, 300);
    })
}