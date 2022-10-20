
window.SURVEY_URL = 'https://www.surveymonkey.com/r/8KCP7P5';
window.SHOW_SURVEY_AFTER_SECONDS = 3*60; // Show the survey after 3min of activity
window.IGNORED_SURVEY_TIMEOUT = 30*24*60*60; // Reshow the survey after 1 month if it was previously ignored
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
    $('#dlg-survey').prepend(
        '<div class="msg">Please click the <b>submit</b> button at the end even if you do not answer all of the questions</div>'
    );

    $('#survey-link').on('click', function() {
        window.tookSurvey = false;
        window.loadingSurvey = false;
        window.showingSurvey = false;
        showSurvey();
    });

    let now = Date.now() / 1000;

    if(localStorage.ignoredSurveyTimestamp && (now - parseFloat(localStorage.ignoredSurveyTimestamp)) > window.IGNORED_SURVEY_TIMEOUT) {
        console.log('Reshowing survey since last time it was ignored');
        window.tookSurvey = false;
        localStorage.removeItem('surveyUrl');
        localStorage.removeItem('ignoredSurveyTimestamp');
        localStorage.activeSeconds = window.SHOW_SURVEY_AFTER_SECONDS;
    }

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

        localStorage.lastActivityTimestamp = now;

        // Track mouse movement and scrolling
        $(window).scroll(updateSurveyActivity);
        $(document).on('mousemove', updateSurveyActivity);
    }
}


function showSurvey() {
    if(localStorage.acceptedCookies && !window.tookSurvey && !window.loadingSurvey) {
        window.loadingSurvey = true;

        $('#iframe-survey').on('load', function() {
            // This is invoked when the survey is completed
            if(window.showingSurvey) {
                localStorage.removeItem('ignoredSurveyTimestamp');
                closeSurvey();
            } else {
                window.showingSurvey = true;
                $('#dlg-survey').css('display', 'block');
            }
        });
        $('#iframe-survey').attr('src', window.SURVEY_URL);

        // This is invoked when the survey is ignored by clicking the exit button
        $("#dlg-survey-close").on("click", function() {
            localStorage.ignoredSurveyTimestamp = Date.now() / 1000;
            closeSurvey();
        });
    }
}

function closeSurvey() {
    console.info('Took or ignored survey');
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