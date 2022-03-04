from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtCore import *
import random
import os
from PyQt5.QtWebEngineWidgets import *
# import Main

t_mood = "Happy"
t_title = "Feather"
t_artist = "Nujabes"
t_url = "https://www.youtube.com/watch?v=jfFTT3iz740"
t_profile = "default"

happy_songs = []
sad_songs = []
current_profile = "Default"
current_mood = "Anger"

# GUI
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

app = QtWidgets.QApplication([])
dlg = uic.loadUi("main_ui.ui")
app.setAttribute(Qt.AA_EnableHighDpiScaling)


class NoSongsFoundError(Exception):
    pass


class Song:
    def __init__(self, mood, title, artist, url):
        self.mood = mood
        self.title = title
        self.artist = artist
        self.url = url


def add_song(mood, title, artist, url, profile):
    """Save a song to the file"""
    # Check if the profile file exists
    try:
        # create file object and write the arguments
        with open("Profiles\\{}.txt".format(profile), 'a') as f:
            f.write("{};{};{};{}\n".format(mood, title, artist, url))
    except FileNotFoundError:
        with open("Profiles\\{}.txt".format(profile), 'w') as f:
            f.write('')
        with open("Profiles\\{}.txt".format(profile), 'a') as f:
            f.write("{};{};{};{}\n".format(mood, title, artist, url))

    dlg.lstPlaylist.addItem("{} - {}".format(artist, title))
    dlg.txtArtistName.setText('')
    dlg.txtSongName.setText('')
    dlg.txtUrl.setText('')


def remove_song():
    """Remove a song from the file"""
    # get the currently selected song from the listbox
    selected_song = str(dlg.lstPlaylist.currentItem().text())
    artist, title = selected_song.split(' - ')

    # change the profile file and delete the song
    with open('Profiles\\{}.txt'.format(current_profile), 'r') as f:
        lines = f.readlines()
    with open('Profiles\\{}.txt'.format(current_profile), 'w') as f:
        for line in lines:
            if '{};{};{}'.format(current_mood, title, artist) not in line.strip():
                f.write(line)

    # Remove the currently selected item from the list
    for item in dlg.lstPlaylist.selectedItems():
        dlg.lstPlaylist.takeItem(dlg.lstPlaylist.row(item))


def add_profile(profile_name):
    """Add a profile to the folder"""
    with open("Profiles\\{}.txt".format(profile_name), 'w') as f:
        pass
    dlg.profiles.addItem(profile_name)
    index = dlg.profiles.findText(profile_name)
    dlg.profiles.setCurrentIndex(index)


def get_song(profile, mood_in):
    """Choose a random song from the profile and mood."""
    songs = []

    with open("Profiles\\{}.txt".format(profile), 'r') as f:
        for line in f:
            mood, title, artist, url = line.strip().split(';')
            new_song = Song(mood, title, artist, url)
            if mood_in == mood:
                # Retrieve attributes using songs[i].mood, title, art, url
                songs.append(new_song)

    if len(songs) == 0:
        raise NoSongsFoundError
    # Use RNG to choose a song from the array
    return songs[random.randint(0, len(songs) - 1)]


def get_song_list():
    """Populate the song list when the selected mood is changed."""
    global current_mood
    current_mood = str(dlg.lstMoods.currentItem().text())
    dlg.lblPlaylist.setText("Playlist - {}".format(current_mood))

    # clear the list to remove the songs from a different mood
    dlg.lstPlaylist.clear()
    # Open the profile file
    with open("Profiles\\{}.txt".format(current_profile), 'r') as f:
        for line in f:
            # Get the song info from each line
            mood, title, artist, url = line.strip().split(';')
            if current_mood.lower() == mood.lower():
                dlg.lstPlaylist.addItem("{} - {}".format(artist, title))


def update_profile():
    """Update the currently selected profile."""
    global current_profile
    current_profile = str(dlg.profiles.currentText())
    print(current_profile)


def parse_current_mood():
    """Get the user's current mood"""

    print("INDEX: " + str(Main.find_Face()))

    index = Main.find_Face()
    out_mood = ''
    if index == 0:
        out_mood = 'Anger'
    elif index == 1:
        out_mood = 'Disgust'
    elif index == 2:
        out_mood = 'Fear'
    elif index == 3:
        out_mood = 'Happy'
    elif index == 4:
        out_mood = 'Sad'
    elif index == 5:
        out_mood = 'Surprise'
    elif index == 6:
        out_mood = 'Neutral'
    return out_mood


def timer_tick():
    """Recheck the mood and then play a new song"""
    user_current_mood = parse_current_mood()

    print("USER CURRENT MOOD:" + user_current_mood)

    try:
        new_song = get_song(current_profile, user_current_mood)
    except NoSongsFoundError:
        return
    dlg.youtubeVideo.setUrl(QUrl(new_song.url))
    dlg.lblCurrentMood.setText("Your current mood is:\n{}".format(user_current_mood))


def gui_init():
    """Call all the GUI initializing functions"""
    # dlg.moodTimer.timeout.connect(test)
    # dlg.moodTimer.start(5000)
    # dlg.youtubeVideo.setUrl(QUrl('https://www.youtube.com/watch?v=D5nsMh2zmXk'))
    # dlg.youtubeVideo.setUrl(QUrl(get_song('Default', 'Happy').url))
    # set the default selected mood to happy
    dlg.lstMoods.setCurrentRow(0)

    # Connect the signals to slots
    dlg.btnExit.clicked.connect(exit)
    dlg.profiles.currentTextChanged.connect(update_profile)
    dlg.lstMoods.currentItemChanged.connect(get_song_list)
    dlg.btnStart.clicked.connect(timer_tick)

    dlg.btnAddSong.clicked.connect(lambda:
                                   add_song(
                                       current_mood,
                                       dlg.txtSongName.text(),
                                       dlg.txtArtistName.text(),
                                       dlg.txtUrl.text(),
                                       current_profile
                                   ))

    dlg.btnAddProfile.clicked.connect(lambda:
                                      add_profile(
                                          dlg.txtAddProfile.text()
                                      ))

    dlg.btnRemove.clicked.connect(remove_song)

    # load font database
    QtGui.QFontDatabase().addApplicationFont('Korean_Calligraphy.ttf')

    # CSS
    app.styleSheet = """
    
    
    QPushButton {
        border-radius: 3px;
        padding:0.3em 1.2em;
        margin:0 0.3em 0.3em 0;
        text-decoration:none;
        font-family:'Roboto',sans-serif;
        font-weight:300;
        color:#FFFFFF;
        background-color:#3500D3;
        text-align:center;
    }

    QMainWindow {
        background-color: #0C0032;
    }

    QLabel {
        color: #ffffff;
        font-family: "Korean Calligraphy";
    }

    QListWidget {
        background-color: #282828;
        color: #FFFFFF;
    }
    
    QLineEdit{
        background-color: #282828;
        color: #FFFFFF;
    }
    
    QComboBox {
        background-color: #282828;
        color: #FFFFFF;
    }
    

    
    """

    # Add all of the moods to the mood list
    dlg.lstMoods.addItem("Anger")
    dlg.lstMoods.addItem("Disgust")
    dlg.lstMoods.addItem("Fear")
    dlg.lstMoods.addItem("Happy")
    dlg.lstMoods.addItem("Sad")
    dlg.lstMoods.addItem("Surprise")
    dlg.lstMoods.addItem("Neutral")

    # Populate the profile combobox
    # get the list of profiles by finding all of the files in the profiles folder
    profile_list = os.listdir('Profiles')
    # loop through all of the profile names and replace the .txt with nothing
    for i, profile in enumerate(profile_list[:]):
        profile_list[i] = profile_list[i].replace('.txt', '')
    dlg.profiles.addItems(profile_list)

    default_index = dlg.profiles.findText('Default')
    dlg.profiles.setCurrentIndex(default_index)


if __name__ == "__main__":
    mood_timer = QTimer()
    mood_timer.timeout.connect(timer_tick)
    mood_timer.start(60000*5)

    gui_init()
    app.setStyleSheet(app.styleSheet)

    dlg.show()
    app.exec()
